/**************************************************************************/
/*  editor_spin_slider.cpp                                                */
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

#include "editor_spin_slider.h"

#include "core/input/input.h"
#include "core/math/expression.h"
#include "core/os/keyboard.h"
#include "editor/settings/editor_settings.h"
#include "editor/themes/editor_scale.h"
#include "scene/gui/line_edit.h"
#include "scene/gui/texture_rect.h"
#include "scene/theme/theme_db.h"

String EditorSpinSlider::get_tooltip(const Point2 &p_pos) const {
	if (read_only || hide_control) {
		return get_text_value() + suffix;
	}

	if (up_button_hovered || down_button_hovered) {
		return "";
	}

	String value = get_text_value() + suffix;
	if (grabber->is_visible()) {
		Key key = (OS::get_singleton()->has_feature("macos") || OS::get_singleton()->has_feature("web_macos") || OS::get_singleton()->has_feature("web_ios")) ? Key::META : Key::CTRL;
		if (!editing_integer) {
			value += "\n\n" + vformat(TTR("Hold %s to round to integers."), find_keycode_name(key));
		}
		value += "\n" + TTR("Hold Shift for more precise changes.");
	}
	return value;
}

String EditorSpinSlider::get_text_value() const {
	return TS->format_number(editing_integer ? itos(get_value()) : rtos(get_value()));
}

void EditorSpinSlider::_update_grabbing_speed() {
	grabbing_spinner_speed = editing_integer ? EDITOR_GET("interface/inspector/integer_drag_speed") : EDITOR_GET("interface/inspector/float_drag_speed");
}

bool EditorSpinSlider::_update_buttons(const Point2 &p_pos, MouseButton p_button, bool p_pressed) {
	bool rtl = is_layout_rtl();
	bool up_hovered = false;
	bool down_hovered = false;

	// Check if the mouse is over the up or down buttons.
	if ((!rtl && p_pos.x > buttons_offset.x) || (rtl && p_pos.x < buttons_offset.x)) {
		if (p_pos.y <= buttons_offset.y && p_pos.y >= 0) {
			up_hovered = true;
		} else if (p_pos.y > buttons_offset.y && p_pos.y <= get_size().height) {
			down_hovered = true;
		}
	}

	// Check if a button has been hovered or unhovered.
	if (up_hovered != up_button_hovered || down_hovered != down_button_hovered) {
		up_button_hovered = up_hovered;
		down_button_hovered = down_hovered;

		// Redraw if mouse motion and no button is held.
		if (p_button == MouseButton::NONE && !up_button_held && !down_button_held) {
			queue_redraw();
		}
	}

	if (p_button == MouseButton::NONE) {
		return false;
	}

	bool accepted = false;
	// Check if the mouse button has been pressed or released and update the held button.
	if (p_pressed) {
		// Set the held button state if a button has been pressed.
		if (up_button_hovered) {
			up_button_held = true;
			accepted = true;
		} else if (down_button_hovered) {
			down_button_held = true;
			accepted = true;
		}
		held_button_index = p_button;
	} else {
		// Check if the held button has been released and update the value.
		if ((up_button_held || down_button_held) && held_button_index == p_button) {
			if (up_button_hovered && up_button_held) {
				set_value((p_button == MouseButton::RIGHT && get_value() < get_max()) ? get_max() : get_value() + get_step());
				emit_signal("updown_pressed");
				accepted = true;
			} else if (down_button_hovered && down_button_held) {
				set_value((p_button == MouseButton::RIGHT && get_value() > get_min()) ? get_min() : get_value() - get_step());
				emit_signal("updown_pressed");
				accepted = true;
			}
		}

		up_button_held = false;
		down_button_held = false;
		held_button_index = MouseButton::NONE;
	}

	if (accepted) {
		accept_event();
	}

	queue_redraw();
	return accepted;
}

void EditorSpinSlider::gui_input(const Ref<InputEvent> &p_event) {
	ERR_FAIL_COND(p_event.is_null());

	if (read_only) {
		return;
	}

	ControlState control_state = _get_control_state();

	Ref<InputEventMouseButton> mb = p_event;
	if (mb.is_valid()) {
		MouseButton button_index = mb->get_button_index();
		if (button_index == MouseButton::LEFT) {
			if (control_state == CONTROL_STATE_USING_ARROWS && _update_buttons(mb->get_position(), MouseButton::LEFT, mb->is_pressed())) {
				return;
			}

			if (mb->is_pressed()) {
				if (!up_button_hovered && !down_button_hovered) {
					_grab_start();
				}
			} else {
				_grab_end();
			}
		} else if (button_index == MouseButton::RIGHT) {
			if (mb->is_pressed()) {
				if (grabbing_grabber || grabbing_spinner) {
					_grab_end();
					set_value(pre_grab_value);
				} else {
					if (control_state == CONTROL_STATE_USING_ARROWS && _update_buttons(mb->get_position(), MouseButton::RIGHT, mb->is_pressed())) {
						return;
					}
				}
			} else {
				if (control_state == CONTROL_STATE_USING_ARROWS && _update_buttons(mb->get_position(), MouseButton::RIGHT, mb->is_pressed())) {
					return;
				}
			}
		}
	}

	Ref<InputEventMouseMotion> mm = p_event;
	if (mm.is_valid()) {
		if (grabbing_spinner_attempt) {
			double diff_x = mm->get_relative().x;
			if (mm->is_shift_pressed() && grabbing_spinner) {
				diff_x *= 0.1;
			}
			grabbing_spinner_dist_cache += diff_x * grabbing_spinner_speed;

			if (!grabbing_spinner && Math::abs(grabbing_spinner_dist_cache) > 4 * grabbing_spinner_speed * EDSCALE) {
				Input::get_singleton()->set_mouse_mode(Input::MOUSE_MODE_CAPTURED);
				grabbing_spinner = true;
			}

			if (grabbing_spinner) {
				// Don't make the user scroll all the way back to 'in range' if they went off the end.
				if (pre_grab_value < get_min() && !is_lesser_allowed()) {
					pre_grab_value = get_min();
				}
				if (pre_grab_value > get_max() && !is_greater_allowed()) {
					pre_grab_value = get_max();
				}

				double new_value = pre_grab_value + (is_layout_rtl() ? -get_step() : get_step()) * grabbing_spinner_dist_cache;
				set_value((mm->is_command_or_control_pressed() && !editing_integer) ? Math::round(new_value) : new_value);
			}
		} else {
			if (control_state == CONTROL_STATE_USING_ARROWS) {
				_update_buttons(mm->get_position());
			}
		}
	}

	Ref<InputEventKey> k = p_event;
	if (k.is_valid() && k->is_pressed()) {
		if (k->is_action("ui_accept", true)) {
			_focus_entered();
		} else if (is_grabbing()) {
			if (k->is_action("ui_cancel", true)) {
				_grab_end();
				set_value(pre_grab_value);
			}
			accept_event();
		}
	}
}

void EditorSpinSlider::_grab_start() {
	grabbing_spinner_attempt = true;
	grabbing_spinner_dist_cache = 0;
	pre_grab_value = get_value();
	grabbing_spinner = false;
	grabbing_spinner_mouse_pos = get_global_mouse_position();
	emit_signal("grabbed");
}

void EditorSpinSlider::_grab_end() {
	if (grabbing_spinner_attempt) {
		if (grabbing_spinner) {
			Input::get_singleton()->set_mouse_mode(Input::MOUSE_MODE_HIDDEN);
			Input::get_singleton()->warp_mouse(grabbing_spinner_mouse_pos);
			Input::get_singleton()->set_mouse_mode(Input::MOUSE_MODE_VISIBLE);
			mouse_over_grabber = true;
			queue_redraw();
			grabbing_spinner = false;
			emit_signal("ungrabbed");
		} else {
			_focus_entered(true);
		}

		grabbing_spinner_attempt = false;
	}

	if (grabbing_grabber) {
		grabbing_grabber = false;
		warping_mouse = false;
		emit_signal("ungrabbed");
	}
}

void EditorSpinSlider::_grabber_gui_input(const Ref<InputEvent> &p_event) {
	if (read_only) {
		return;
	}

	Ref<InputEventMouseButton> mb = p_event;

	if (mb.is_valid()) {
		MouseButton button_index = mb->get_button_index();
		if (button_index == MouseButton::LEFT) {
			if (mb->is_pressed()) {
				grabbing_grabber = true;
				pre_grab_value = get_value();
				if (!warping_mouse) {
					grabbing_ratio = get_as_ratio();
					Point2 grabbing_position = grabber->get_transform().xform(mb->get_position());
					grabbing_from = grabbing_position.x;
					warping_mouse_offset = grabber->get_transform().xform(grabber->get_size() * 0.5) - grabbing_position;
				}
				grab_focus(true);
				accept_event();
				emit_signal("grabbed");
			} else {
				if (grabbing_grabber) {
					grabbing_grabber = false;
					if (!mouse_over_grabber) {
						queue_redraw();
					}
				}
				warping_mouse = false;
				accept_event();
				emit_signal("ungrabbed");
			}
		}

		// Change value with mouse wheel or cancel with right-click while dragging the grabber.
		if (grabbing_grabber && mb->is_pressed()) {
			if (button_index == MouseButton::RIGHT) {
				grabbing_grabber = false;
				warping_mouse = false;
				set_value(pre_grab_value);
				accept_event();
				emit_signal("ungrabbed");
			} else if (button_index == MouseButton::WHEEL_UP || button_index == MouseButton::WHEEL_DOWN) {
				set_value((button_index == MouseButton::WHEEL_UP) ? (get_value() + get_step()) : (get_value() - get_step()));
				warping_mouse = true;
				set_process_internal(true);
				accept_event();
			}
		}
	}

	// Change the value by dragging the grabber.
	Ref<InputEventMouseMotion> mm = p_event;
	if (mm.is_valid() && grabbing_grabber) {
		if (warping_mouse) {
			return;
		}

		float scale_x = get_global_transform_with_canvas().get_scale().x;
		ERR_FAIL_COND(Math::is_zero_approx(scale_x));
		float grabbing_ofs = (grabber->get_transform().xform(mm->get_position()).x - grabbing_from) / float(grabber_range) / scale_x;
		set_as_ratio(grabbing_ratio + (is_layout_rtl() ? -grabbing_ofs : grabbing_ofs));
		queue_redraw();
	}
}

void EditorSpinSlider::_value_input_gui_input(const Ref<InputEvent> &p_event) {
	Ref<InputEventKey> k = p_event;
	if (k.is_valid() && k->is_pressed() && !read_only) {
		Key code = k->get_keycode();

		switch (code) {
			case Key::UP:
			case Key::DOWN: {
				double step = get_step();
				if (step < 1) {
					double divisor = 1.0 / step;

					if (std::trunc(divisor) == divisor) {
						step = 1.0;
					}
				}

				if (k->is_command_or_control_pressed()) {
					step *= 100.0;
				} else if (k->is_shift_pressed()) {
					step *= 10.0;
				} else if (k->is_alt_pressed()) {
					step *= 0.1;
				}

				_evaluate_input_text();

				double last_value = get_value();
				if (code == Key::DOWN) {
					step *= -1;
				}
				set_value(last_value + step);

				value_input_dirty = true;
				set_process_internal(true);
			} break;
			case Key::ESCAPE: {
				value_input_closed_frame = Engine::get_singleton()->get_frames_drawn();
				if (value_input) {
					value_input->hide();
				}
			} break;
			default:
				break;
		}
	}
}

void EditorSpinSlider::_update_value_input_stylebox() {
	if (!value_input) {
		return;
	}

	Side side;
	int suffix_width = 0;
	if (is_layout_rtl()) {
		side = SIDE_RIGHT;
		if (!suffix.is_empty()) {
			suffix_width = theme_cache.font->get_string_size(suffix + U"\u2009", HORIZONTAL_ALIGNMENT_LEFT, -1, theme_cache.font_size).width;
		}
	} else {
		side = SIDE_LEFT;
	}

	Ref<StyleBox> stylebox = theme_cache.normal_style->duplicate();
	int margin = stylebox->get_margin(side);
	int begin_ofs = 0;

	// Calculate text begin offset including label width and separation.
	if (label.is_empty()) {
		begin_ofs = 8 * EDSCALE;
	} else {
		begin_ofs = margin * 2 + theme_cache.font->get_string_size(label, HORIZONTAL_ALIGNMENT_LEFT, -1, theme_cache.font_size).width;
	}

	stylebox->set_content_margin(side, margin + begin_ofs + suffix_width);
	value_input->add_theme_style_override(CoreStringName(normal), stylebox);
}

void EditorSpinSlider::_draw_spin_slider() {
	buttons_offset = Point2(-1, -1);
	_shape();

	RID ci = get_canvas_item();
	bool rtl = is_layout_rtl();
	Vector2 size = get_size();
	bool draw_label = !label.is_empty();
	Ref<StyleBox> style = read_only ? theme_cache.read_only_style : theme_cache.normal_style;
	Size2 style_min_size = style->get_minimum_size();

	if (!flat) {
		draw_style_box(style, Rect2(Vector2(), size));
	}

	int margin = rtl ? style->get_margin(SIDE_RIGHT) : style->get_margin(SIDE_LEFT);
	int sep = 0;
	int ofs = 0;
	int label_width = 0;
	int vofs = (size.height - TS->shaped_text_get_size(text_rid).y) / 2 + TS->shaped_text_get_ascent(text_rid);
	Color fc = read_only ? theme_cache.font_uneditable_color : theme_cache.font_color;

	if (draw_label) {
		// Use double margin for separation, once for the label stylebox to have equal margins and once for the separation between the stylebox and the value when flat is enabled.
		sep = margin * 2;
		label_width = theme_cache.font->get_string_size(label, HORIZONTAL_ALIGNMENT_LEFT, -1, theme_cache.font_size).width;

		if (flat) {
			if (rtl) {
				draw_style_box(theme_cache.label_bg_style, Rect2(Vector2(size.width - (style->get_offset().x * 2 + label_width), 0), Vector2(style->get_offset().x * 2 + label_width, size.height)));
			} else {
				draw_style_box(theme_cache.label_bg_style, Rect2(Vector2(), Vector2(style->get_offset().x * 2 + label_width, size.height)));
			}
		}
	} else {
		// Offset the text by 8 pixels when the label is empty.
		ofs = 8 * EDSCALE;
	}

	if (has_focus(true)) {
		draw_style_box(theme_cache.focus_style, Rect2(Vector2(), size));
	}

	if (draw_label) {
		Color lc = read_only ? theme_cache.read_only_label_color : theme_cache.label_color;

		if (rtl) {
			draw_string(theme_cache.font, Vector2(size.width - style->get_margin(SIDE_RIGHT) - label_width, vofs), label, HORIZONTAL_ALIGNMENT_RIGHT, -1, theme_cache.font_size, lc * Color(1, 1, 1, 0.5));
		} else {
			draw_string(theme_cache.font, Vector2(style->get_margin(SIDE_LEFT), vofs), label, HORIZONTAL_ALIGNMENT_LEFT, -1, theme_cache.font_size, lc * Color(1, 1, 1, 0.5));
		}
	}

	int number_width = size.width - style_min_size.width - label_width - sep - ofs;
	int text_start = rtl ? margin : margin + label_width + sep + ofs;
	int text_ofs = rtl ? text_start + (number_width - TS->shaped_text_get_width(text_rid)) : text_start;
	int v_size = TS->shaped_text_get_glyph_count(text_rid);
	const Glyph *glyphs = TS->shaped_text_get_glyphs(text_rid);
	for (int i = 0; i < v_size; i++) {
		for (int j = 0; j < glyphs[i].repeat; j++) {
			if (text_ofs >= text_start && (text_ofs + glyphs[i].advance) <= (text_start + number_width)) {
				Color color = fc;
				if (glyphs[i].start >= suffix_start) {
					color.a *= 0.4;
				}
				if (glyphs[i].font_rid != RID()) {
					TS->font_draw_glyph(glyphs[i].font_rid, ci, glyphs[i].font_size, Vector2(glyphs[i].x_off + text_ofs, glyphs[i].y_off + vofs), glyphs[i].index, color);
				} else if (((glyphs[i].flags & TextServer::GRAPHEME_IS_VIRTUAL) != TextServer::GRAPHEME_IS_VIRTUAL) && ((glyphs[i].flags & TextServer::GRAPHEME_IS_EMBEDDED_OBJECT) != TextServer::GRAPHEME_IS_EMBEDDED_OBJECT)) {
					TS->draw_hex_code_box(ci, glyphs[i].font_size, Vector2(glyphs[i].x_off + text_ofs, glyphs[i].y_off + vofs), glyphs[i].index, color);
				}
			}
			text_ofs += glyphs[i].advance;
		}
	}

	switch (_get_control_state()) {
		case CONTROL_STATE_USING_ARROWS: {
			int icon_sep = MAX(0, theme_cache.updown_v_separation);
			int icon_max_height = (size.height - icon_sep - style_min_size.height) / 2;
			Size2 up_icon_ms = theme_cache.up_icon->get_size();
			Size2 down_icon_ms = theme_cache.down_icon->get_size();
			Size2 up_icon_size = Size2(up_icon_ms.width, MAX(up_icon_ms.height, icon_max_height));
			Size2 down_icon_size = Size2(down_icon_ms.width, MAX(down_icon_ms.height, icon_max_height));

			int icon_x_pos;
			if (rtl) {
				icon_x_pos = margin;
				buttons_offset.x = icon_x_pos + MAX(up_icon_size.width, down_icon_size.width);
			} else {
				icon_x_pos = size.width - MAX(up_icon_size.width, down_icon_size.width) - margin;
				buttons_offset.x = icon_x_pos;
			}

			int half_sep = icon_sep / 2;
			buttons_offset.y = size.height * 0.5;
			Point2 up_icon_pos = Point2(icon_x_pos, buttons_offset.y - up_icon_size.height - half_sep);
			Point2 down_icon_pos = Point2(icon_x_pos, buttons_offset.y + half_sep);
			bool up_button_disabled = get_value() == get_max() && !is_greater_allowed();
			bool down_button_disabled = get_value() == get_min() && !is_lesser_allowed();
			float up_icon_alpha = up_button_disabled ? 0.4 : 0.8;
			float down_icon_alpha = down_button_disabled ? 0.4 : 0.8;

			if (read_only) {
				up_icon_alpha = 0.4;
				down_icon_alpha = 0.4;
			} else {
				if (up_button_hovered && !up_button_disabled) {
					up_icon_alpha = 1.0;
					draw_style_box(up_button_held ? theme_cache.updown_pressed_style : theme_cache.updown_hovered_style, Rect2(up_icon_pos, up_icon_size));
				} else if (down_button_hovered && !down_button_disabled) {
					down_icon_alpha = 1.0;
					draw_style_box(down_button_held ? theme_cache.updown_pressed_style : theme_cache.updown_hovered_style, Rect2(down_icon_pos, down_icon_size));
				}
			}

			up_icon_pos.y += MAX(0, (up_icon_size.height - up_icon_ms.height) / 2);
			down_icon_pos.y += MAX(0, (down_icon_size.height - down_icon_ms.height) / 2);
			draw_texture(theme_cache.up_icon, up_icon_pos, Color(1, 1, 1, up_icon_alpha));
			draw_texture(theme_cache.down_icon, down_icon_pos, Color(1, 1, 1, down_icon_alpha));
		} break;

		case CONTROL_STATE_USING_SLIDER: {
			int slider_bg_height = MAX(2, 2 * EDSCALE);
			int grabber_size = slider_bg_height + 2;
			int gofs = slider_bg_height * 0.5;
			int svofs = ((size.height + vofs) * 0.5) - gofs;

			// Calculate begin offset and grabber position.
			int begin = MAX(margin, slider_bg_height);
			int width = size.width - MAX(style_min_size.width, grabber_size);
			int grabber_pos = (rtl ? (1 - get_as_ratio()) : get_as_ratio()) * width;
			Color c = fc;

			// Draw the horizontal slider's background.
			c.a = 0.2;
			draw_rect(Rect2(begin, svofs, width, slider_bg_height), c);

			c.a = 0.45;
			// Draw the horizontal slider's filled part.
			if (rtl) {
				draw_rect(Rect2(grabber_pos + begin, svofs, width - grabber_pos, slider_bg_height), c);
			} else {
				draw_rect(Rect2(begin, svofs, grabber_pos, slider_bg_height), c);
			}

			bool display_grabber = false;
			if (!read_only && !grabbing_spinner && !(value_input && value_input->is_visible())) {
				display_grabber = (grabbing_grabber || mouse_over_spin || mouse_over_grabber);
			}

			if (grabber->is_visible() != display_grabber) {
				grabber->set_visible(display_grabber);
			}

			Rect2 grabber_rect = Rect2(grabber_pos + begin - slider_bg_height, svofs - gofs, grabber_size, grabber_size);
			grabbing_spinner_mouse_pos = get_global_position() + grabber_rect.get_center();

			// Draw the slider grabber.
			if (display_grabber) {
				Ref<Texture2D> grabber_tex = (mouse_over_grabber || grabbing_grabber) ? theme_cache.grabber_highlight_icon : theme_cache.grabber_icon;
				if (grabber->get_texture() != grabber_tex) {
					grabber->set_texture(grabber_tex);
				}

				grabber->reset_size();
				grabber->set_position(grabber_rect.get_center() - grabber->get_size() * 0.5);

				if (warping_mouse && !warping_mouse_queue_disable) {
					Input::get_singleton()->set_mouse_mode(Input::MOUSE_MODE_HIDDEN);
					Input::get_singleton()->warp_mouse(grabbing_spinner_mouse_pos - warping_mouse_offset);
					Input::get_singleton()->set_mouse_mode(Input::MOUSE_MODE_VISIBLE);
					warping_mouse_queue_disable = true;
					set_process_internal(true);
				}

				grabber_range = width;
			} else {
				c.a = 0.9;
				draw_rect(grabber_rect, c);
			}
		} break;

		case CONTROL_STATE_HIDDEN: {
			// Do nothing.
		} break;
	}
}

EditorSpinSlider::ControlState EditorSpinSlider::_get_control_state() const {
	return hide_control ? CONTROL_STATE_HIDDEN : state;
}

void EditorSpinSlider::_update_control_state() {
	state = (editing_integer && !integer_prefer_slider) || (!editing_integer && float_prefer_arrows)
			? CONTROL_STATE_USING_ARROWS
			: CONTROL_STATE_USING_SLIDER;
}

void EditorSpinSlider::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE: {
			_update_grabbing_speed();
			_update_value_input_stylebox();
		} break;

		case NOTIFICATION_LAYOUT_DIRECTION_CHANGED:
		case NOTIFICATION_TRANSLATION_CHANGED:
		case NOTIFICATION_THEME_CHANGED: {
			text_dirty = true;
			_update_value_input_stylebox();
		} break;

		case NOTIFICATION_INTERNAL_PROCESS: {
			if (value_input_dirty) {
				value_input_dirty = false;
				value_input->set_text(get_text_value());
			}

			if (warping_mouse_queue_disable) {
				warping_mouse_queue_disable = false;
			} else {
				warping_mouse = false;
				set_process_internal(false);
			}
		} break;

		case NOTIFICATION_DRAW: {
			_draw_spin_slider();
		} break;

		case NOTIFICATION_WM_WINDOW_FOCUS_IN:
		case NOTIFICATION_WM_WINDOW_FOCUS_OUT:
		case NOTIFICATION_WM_CLOSE_REQUEST:
		case NOTIFICATION_EXIT_TREE: {
			if (grabbing_spinner) {
				grabber->hide();
				Input::get_singleton()->set_mouse_mode(Input::MOUSE_MODE_VISIBLE);
				Input::get_singleton()->warp_mouse(grabbing_spinner_mouse_pos);
				grabbing_spinner = false;
				grabbing_spinner_attempt = false;
			}
		} break;

		case NOTIFICATION_MOUSE_ENTER: {
			mouse_over_spin = true;

			if (_get_control_state() == CONTROL_STATE_USING_ARROWS) {
				BitField<MouseButtonMask> mb_mask = Input::get_singleton()->get_mouse_button_mask();
				if (mb_mask.has_flag(MouseButtonMask::LEFT) || mb_mask.has_flag(MouseButtonMask::RIGHT)) {
					return;
				}

				_update_buttons(get_local_mouse_position());
			} else {
				queue_redraw();
			}
		} break;

		case NOTIFICATION_MOUSE_EXIT: {
			mouse_over_spin = false;
			up_button_hovered = false;
			down_button_hovered = false;

			if (!up_button_held && !down_button_held) {
				queue_redraw();
			}
		} break;

		case NOTIFICATION_FOCUS_ENTER: {
			if ((Input::get_singleton()->is_action_pressed("ui_focus_next") || Input::get_singleton()->is_action_pressed("ui_focus_prev")) && value_input_closed_frame != Engine::get_singleton()->get_frames_drawn()) {
				_focus_entered();
			}
			value_input_closed_frame = 0;
		} break;

		case EditorSettings::NOTIFICATION_EDITOR_SETTINGS_CHANGED: {
			if (EditorSettings::get_singleton()->check_changed_settings_in_group("interface/inspector")) {
				_update_grabbing_speed();
			}
		} break;
	}
}

LineEdit *EditorSpinSlider::get_line_edit() {
	_ensure_value_input();
	return value_input;
}

Size2 EditorSpinSlider::get_minimum_size() const {
	Size2 ms = theme_cache.normal_style->get_minimum_size();
	int text_height = MAX(TS->shaped_text_get_size(text_rid).height, theme_cache.font->get_height(theme_cache.font_size));

	if (_get_control_state() == CONTROL_STATE_USING_ARROWS) {
		int min_height = theme_cache.up_icon->get_height() + theme_cache.down_icon->get_height() + MAX(theme_cache.updown_v_separation, 0);
		ms.height += MAX(min_height, text_height);
	} else {
		ms.height += text_height;
	}
	return ms;
}

#ifndef DISABLE_DEPRECATED
void EditorSpinSlider::set_hide_slider(bool p_hide) {
	set_hide_control(p_hide);
}

bool EditorSpinSlider::is_hiding_slider() const {
	return hide_control;
}
#endif

void EditorSpinSlider::set_hide_control(bool p_hide) {
	if (hide_control == p_hide) {
		return;
	}
	hide_control = p_hide;
	mouse_over_grabber = false;

	if (hide_control && grabber->is_visible()) {
		grabber->hide();
	}

	queue_redraw();
}

bool EditorSpinSlider::is_hiding_control() const {
	return hide_control;
}

void EditorSpinSlider::set_editing_integer(bool p_editing_integer) {
	if (p_editing_integer == editing_integer) {
		return;
	}

	editing_integer = p_editing_integer;
	if (is_inside_tree()) {
		_update_grabbing_speed();
	}
	_update_control_state();

	if (!hide_control) {
		queue_redraw();
	}
}

bool EditorSpinSlider::is_editing_integer() const {
	return editing_integer;
}

void EditorSpinSlider::set_integer_prefer_slider(bool p_enable) {
	if (integer_prefer_slider == p_enable) {
		return;
	}
	integer_prefer_slider = p_enable;
	mouse_over_grabber = false;
	_update_control_state();

	if (!hide_control && editing_integer) {
		queue_redraw();
	}
}

bool EditorSpinSlider::is_integer_preferring_slider() const {
	return integer_prefer_slider;
}

void EditorSpinSlider::set_float_prefer_arrows(bool p_enable) {
	if (float_prefer_arrows == p_enable) {
		return;
	}
	float_prefer_arrows = p_enable;
	mouse_over_grabber = false;
	_update_control_state();

	if (!hide_control && !editing_integer) {
		queue_redraw();
	}
}

bool EditorSpinSlider::is_float_preferring_arrows() const {
	return float_prefer_arrows;
}

void EditorSpinSlider::set_label(const String &p_label) {
	if (label == p_label) {
		return;
	}
	label = p_label;
	_update_value_input_stylebox();
	queue_redraw();
}

String EditorSpinSlider::get_label() const {
	return label;
}

void EditorSpinSlider::set_suffix(const String &p_suffix) {
	if (suffix == p_suffix) {
		return;
	}
	suffix = p_suffix;
	text_dirty = true;
	_update_value_input_stylebox();
	queue_redraw();
}

String EditorSpinSlider::get_suffix() const {
	return suffix;
}

void EditorSpinSlider::_evaluate_input_text() {
	Ref<Expression> expr;
	expr.instantiate();

	// Convert commas ',' to dots '.' for French/German etc. keyboard layouts.
	String text = value_input->get_text().replace_char(',', '.');
	text = text.replace_char(';', ',');
	text = TS->parse_number(text);

	Error err = expr->parse(text);
	if (err != OK) {
		// If the expression failed try without converting commas to dots - they might have been for parameter separation.
		text = value_input->get_text();
		text = TS->parse_number(text);

		err = expr->parse(text);
		if (err != OK) {
			return;
		}
	}

	Variant v = expr->execute(Array(), nullptr, false, true);
	if (v.get_type() == Variant::NIL) {
		return;
	}
	set_value(v);
}

//text_submitted signal
void EditorSpinSlider::_value_input_submitted(const String &p_text) {
	value_input_closed_frame = Engine::get_singleton()->get_frames_drawn();
	if (value_input) {
		value_input->hide();
	}
}

//modal_closed signal
void EditorSpinSlider::_value_input_closed() {
	_evaluate_input_text();
	value_input_closed_frame = Engine::get_singleton()->get_frames_drawn();
}

//focus_exited signal
void EditorSpinSlider::_value_focus_exited() {
	// discontinue because the focus_exit was caused by right-click context menu
	if (value_input->is_menu_visible()) {
		return;
	}

	if (read_only) {
		// Spin slider has become read only while it was being edited.
		return;
	}

	_evaluate_input_text();
	// focus is not on the same element after the value_input was exited
	// -> focus is on next element
	// -> TAB was pressed
	// -> modal_close was not called
	// -> need to close/hide manually
	if (!is_visible_in_tree() || value_input_closed_frame != Engine::get_singleton()->get_frames_drawn()) {
		// Hidden or something else took focus.
		if (value_input) {
			value_input->hide();
		}
	} else {
		// Enter or Esc was pressed.
		grab_focus();
	}

	emit_signal("value_focus_exited");
}

void EditorSpinSlider::_shape() {
	String text_value = get_text_value();
	if (!text_dirty && prev_value == text_value) {
		return;
	}
	prev_value = text_value;
	TS->shaped_text_clear(text_rid);
	suffix_start = text_value.length();

	if (!suffix.is_empty()) {
		text_value = text_value + U"\u2009" + suffix;
	}

	TS->shaped_text_add_string(text_rid, text_value, theme_cache.font->get_rids(), theme_cache.font_size, theme_cache.font->get_opentype_features());
}

void EditorSpinSlider::_grabber_mouse_entered() {
	mouse_over_grabber = true;
	queue_redraw();
}

void EditorSpinSlider::_grabber_mouse_exited() {
	mouse_over_grabber = false;
	queue_redraw();
}

void EditorSpinSlider::set_read_only(bool p_enable) {
	read_only = p_enable;
	if (read_only && value_input && value_input->is_inside_tree()) {
		value_input->release_focus();
	}

	queue_redraw();
}

bool EditorSpinSlider::is_read_only() const {
	return read_only;
}

void EditorSpinSlider::set_flat(bool p_enable) {
	if (flat == p_enable) {
		return;
	}

	flat = p_enable;
	queue_redraw();
}

bool EditorSpinSlider::is_flat() const {
	return flat;
}

bool EditorSpinSlider::is_grabbing() const {
	return grabbing_grabber || grabbing_spinner;
}

void EditorSpinSlider::_focus_entered(bool p_hide_focus) {
	if (read_only) {
		return;
	}

	_ensure_value_input();
	value_input->set_text(get_text_value());
	value_input->set_size(get_size());
	value_input->set_focus_next(find_next_valid_focus()->get_path());
	value_input->set_focus_previous(find_prev_valid_focus()->get_path());
	callable_mp((CanvasItem *)value_input, &CanvasItem::show).call_deferred();
	callable_mp((Control *)value_input, &Control::grab_focus).call_deferred(p_hide_focus);
	callable_mp(value_input, &LineEdit ::select_all).call_deferred();
	emit_signal("value_focus_entered");
}

void EditorSpinSlider::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_label", "label"), &EditorSpinSlider::set_label);
	ClassDB::bind_method(D_METHOD("get_label"), &EditorSpinSlider::get_label);

	ClassDB::bind_method(D_METHOD("set_suffix", "suffix"), &EditorSpinSlider::set_suffix);
	ClassDB::bind_method(D_METHOD("get_suffix"), &EditorSpinSlider::get_suffix);

	ClassDB::bind_method(D_METHOD("set_read_only", "read_only"), &EditorSpinSlider::set_read_only);
	ClassDB::bind_method(D_METHOD("is_read_only"), &EditorSpinSlider::is_read_only);

	ClassDB::bind_method(D_METHOD("set_flat", "flat"), &EditorSpinSlider::set_flat);
	ClassDB::bind_method(D_METHOD("is_flat"), &EditorSpinSlider::is_flat);

#ifndef DISABLE_DEPRECATED
	ClassDB::bind_method(D_METHOD("set_hide_slider", "hide_slider"), &EditorSpinSlider::set_hide_slider);
	ClassDB::bind_method(D_METHOD("is_hiding_slider"), &EditorSpinSlider::is_hiding_slider);
#endif

	ClassDB::bind_method(D_METHOD("set_editing_integer", "editing_integer"), &EditorSpinSlider::set_editing_integer);
	ClassDB::bind_method(D_METHOD("is_editing_integer"), &EditorSpinSlider::is_editing_integer);

	ClassDB::bind_method(D_METHOD("set_hide_control", "hide"), &EditorSpinSlider::set_hide_control);
	ClassDB::bind_method(D_METHOD("is_hiding_control"), &EditorSpinSlider::is_hiding_control);

	ClassDB::bind_method(D_METHOD("set_integer_prefer_slider", "integer_prefer_slider"), &EditorSpinSlider::set_integer_prefer_slider);
	ClassDB::bind_method(D_METHOD("is_integer_preferring_slider"), &EditorSpinSlider::is_integer_preferring_slider);

	ClassDB::bind_method(D_METHOD("set_float_prefer_arrows", "float_prefer_arrows"), &EditorSpinSlider::set_float_prefer_arrows);
	ClassDB::bind_method(D_METHOD("is_float_preferring_arrows"), &EditorSpinSlider::is_float_preferring_arrows);

	ADD_PROPERTY(PropertyInfo(Variant::STRING, "label"), "set_label", "get_label");
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "suffix"), "set_suffix", "get_suffix");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "read_only"), "set_read_only", "is_read_only");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "flat"), "set_flat", "is_flat");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "hide_control"), "set_hide_control", "is_hiding_control");
#ifndef DISABLE_DEPRECATED
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "hide_slider"), "set_hide_slider", "is_hiding_slider");
#endif
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "editing_integer"), "set_editing_integer", "is_editing_integer");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "integer_prefer_slider"), "set_integer_prefer_slider", "is_integer_preferring_slider");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "float_prefer_arrows"), "set_float_prefer_arrows", "is_float_preferring_arrows");

	ADD_SIGNAL(MethodInfo("grabbed"));
	ADD_SIGNAL(MethodInfo("ungrabbed"));
	ADD_SIGNAL(MethodInfo("updown_pressed"));
	ADD_SIGNAL(MethodInfo("value_focus_entered"));
	ADD_SIGNAL(MethodInfo("value_focus_exited"));

	BIND_THEME_ITEM_CUSTOM(Theme::DATA_TYPE_STYLEBOX, EditorSpinSlider, normal_style, "normal");
	BIND_THEME_ITEM_CUSTOM(Theme::DATA_TYPE_STYLEBOX, EditorSpinSlider, read_only_style, "read_only");
	BIND_THEME_ITEM_CUSTOM(Theme::DATA_TYPE_STYLEBOX, EditorSpinSlider, focus_style, "focus");
	BIND_THEME_ITEM_CUSTOM(Theme::DATA_TYPE_STYLEBOX, EditorSpinSlider, label_bg_style, "label_bg");
	BIND_THEME_ITEM_CUSTOM(Theme::DATA_TYPE_STYLEBOX, EditorSpinSlider, updown_hovered_style, "updown_hovered");
	BIND_THEME_ITEM_CUSTOM(Theme::DATA_TYPE_STYLEBOX, EditorSpinSlider, updown_pressed_style, "updown_pressed");

	BIND_THEME_ITEM(Theme::DATA_TYPE_FONT, EditorSpinSlider, font);
	BIND_THEME_ITEM(Theme::DATA_TYPE_FONT_SIZE, EditorSpinSlider, font_size);

	BIND_THEME_ITEM(Theme::DATA_TYPE_CONSTANT, EditorSpinSlider, updown_v_separation);

	BIND_THEME_ITEM(Theme::DATA_TYPE_COLOR, EditorSpinSlider, font_color);
	BIND_THEME_ITEM(Theme::DATA_TYPE_COLOR, EditorSpinSlider, font_uneditable_color);
	BIND_THEME_ITEM(Theme::DATA_TYPE_COLOR, EditorSpinSlider, label_color);
	BIND_THEME_ITEM(Theme::DATA_TYPE_COLOR, EditorSpinSlider, read_only_label_color);

	BIND_THEME_ITEM_CUSTOM(Theme::DATA_TYPE_ICON, EditorSpinSlider, grabber_icon, "grabber");
	BIND_THEME_ITEM_CUSTOM(Theme::DATA_TYPE_ICON, EditorSpinSlider, grabber_highlight_icon, "grabber_highlight");
	BIND_THEME_ITEM_CUSTOM(Theme::DATA_TYPE_ICON, EditorSpinSlider, up_icon, "up_arrow");
	BIND_THEME_ITEM_CUSTOM(Theme::DATA_TYPE_ICON, EditorSpinSlider, down_icon, "down_arrow");
}

void EditorSpinSlider::_ensure_value_input() {
	if (value_input) {
		return;
	}

	value_input = memnew(LineEdit);
	value_input->set_emoji_menu_enabled(false);
	value_input->set_focus_mode(FOCUS_CLICK);
	add_child(value_input);
	value_input->set_anchors_and_offsets_preset(PRESET_FULL_RECT);
	value_input->connect(SceneStringName(hidden), callable_mp(this, &EditorSpinSlider::_value_input_closed));
	value_input->connect(SceneStringName(text_submitted), callable_mp(this, &EditorSpinSlider::_value_input_submitted));
	value_input->connect(SceneStringName(focus_exited), callable_mp(this, &EditorSpinSlider::_value_focus_exited));
	value_input->connect(SceneStringName(gui_input), callable_mp(this, &EditorSpinSlider::_value_input_gui_input));

	if (is_inside_tree()) {
		_update_value_input_stylebox();
	}
}

EditorSpinSlider::EditorSpinSlider() {
	text_rid = TS->create_shaped_text();

	set_focus_mode(FOCUS_ALL);
	grabber = memnew(TextureRect);
	add_child(grabber);
	grabber->hide();
	grabber->set_z_index(1);
	grabber->set_mouse_filter(MOUSE_FILTER_STOP);
	grabber->connect(SceneStringName(mouse_entered), callable_mp(this, &EditorSpinSlider::_grabber_mouse_entered));
	grabber->connect(SceneStringName(mouse_exited), callable_mp(this, &EditorSpinSlider::_grabber_mouse_exited));
	grabber->connect(SceneStringName(gui_input), callable_mp(this, &EditorSpinSlider::_grabber_gui_input));
}

EditorSpinSlider::~EditorSpinSlider() {
	TS->free_rid(text_rid);
}
