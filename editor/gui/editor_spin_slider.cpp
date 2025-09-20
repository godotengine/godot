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
#include "scene/theme/theme_db.h"

String EditorSpinSlider::get_tooltip(const Point2 &p_pos) const {
	if (up_icon_hovered || down_icon_hovered) {
		return "";
	}

	if (!read_only && grabber->is_visible()) {
		Key key = (OS::get_singleton()->has_feature("macos") || OS::get_singleton()->has_feature("web_macos") || OS::get_singleton()->has_feature("web_ios")) ? Key::META : Key::CTRL;
		return TS->format_number(rtos(get_value())) + "\n\n" + vformat(TTR("Hold %s to round to integers.\nHold Shift for more precise changes."), find_keycode_name(key));
	}
	return TS->format_number(rtos(get_value()));
}

String EditorSpinSlider::get_text_value() const {
	return TS->format_number(String::num(get_value(), Math::range_step_decimals(get_step())));
}

void EditorSpinSlider::_update_button_hover_state(const Point2 &p_pos) {
	if (updown_offset.x == -1) {
		up_icon_hovered = false;
		down_icon_hovered = false;
		return;
	}

	bool rtl = is_layout_rtl();
	bool up_hovered = false;
	bool down_hovered = false;
	if ((!rtl && p_pos.x > updown_offset.x) || (rtl && p_pos.x < updown_offset.x)) {
		if (p_pos.y <= updown_offset.y && p_pos.y >= 0) {
			up_hovered = true;
			down_hovered = false;
		} else if (p_pos.y > updown_offset.y && p_pos.y <= get_size().height) {
			down_hovered = true;
			up_hovered = false;
		}
	}

	if (up_hovered != up_icon_hovered || down_hovered != down_icon_hovered) {
		up_icon_hovered = up_hovered;
		down_icon_hovered = down_hovered;
		queue_redraw();
	}
}

void EditorSpinSlider::gui_input(const Ref<InputEvent> &p_event) {
	ERR_FAIL_COND(p_event.is_null());

	if (read_only) {
		return;
	}

	Ref<InputEventMouseButton> mb = p_event;
	if (mb.is_valid()) {
		if (mb->get_button_index() == MouseButton::LEFT) {
			_update_button_hover_state(mb->get_position());

			if (mb->is_pressed()) {
				if (up_icon_hovered) {
					updown_pressed = true;
					set_value(get_value() + get_step());
				} else if (down_icon_hovered) {
					updown_pressed = true;
					set_value(get_value() - get_step());
				}

				if (updown_pressed) {
					emit_signal("updown_pressed");
					queue_redraw();
				} else {
					_grab_start();
				}
			} else {
				if (updown_pressed) {
					updown_pressed = false;
					queue_redraw();
				}

				_grab_end();
			}
		} else if (mb->get_button_index() == MouseButton::RIGHT) {
			if (mb->is_pressed()) {
				if (is_grabbing()) {
					_grab_end();
					set_value(pre_grab_value);
				} else {
					if (up_icon_hovered && get_value() < get_max()) {
						updown_pressed = true;
						set_value(get_max());
					} else if (down_icon_hovered && get_value() > get_min()) {
						updown_pressed = true;
						set_value(get_min());
					}
				}
			} else {
				_update_button_hover_state(mb->get_position());

				if (updown_pressed) {
					updown_pressed = false;
					queue_redraw();
				}
			}
		} else if (mb->get_button_index() == MouseButton::WHEEL_UP || mb->get_button_index() == MouseButton::WHEEL_DOWN) {
			if (grabber->is_visible()) {
				callable_mp((CanvasItem *)this, &CanvasItem::queue_redraw).call_deferred();
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
				set_value(mm->is_command_or_control_pressed() ? Math::round(new_value) : new_value);
			}
		} else if (updown_offset.x != -1) {
			if (!updown_pressed) {
				_update_button_hover_state(mm->get_position());
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
			Input::get_singleton()->set_mouse_mode(Input::MOUSE_MODE_VISIBLE);
			Input::get_singleton()->warp_mouse(grabbing_spinner_mouse_pos);
			queue_redraw();
			grabbing_spinner = false;
			emit_signal("ungrabbed");
		} else {
			_focus_entered();
		}

		grabbing_spinner_attempt = false;
	}

	if (grabbing_grabber) {
		grabbing_grabber = false;
		mousewheel_over_grabber = false;
		emit_signal("ungrabbed");
	}
}

void EditorSpinSlider::_grabber_gui_input(const Ref<InputEvent> &p_event) {
	if (read_only) {
		return;
	}

	Ref<InputEventMouseButton> mb = p_event;

	if (grabbing_grabber) {
		if (mb.is_valid()) {
			if (mb->get_button_index() == MouseButton::WHEEL_UP) {
				set_value(get_value() + get_step());
				mousewheel_over_grabber = true;
				accept_event();
			} else if (mb->get_button_index() == MouseButton::WHEEL_DOWN) {
				set_value(get_value() - get_step());
				mousewheel_over_grabber = true;
				accept_event();
			}
		}
	}

	if (mb.is_valid() && mb->get_button_index() == MouseButton::LEFT) {
		if (mb->is_pressed()) {
			grabbing_grabber = true;
			pre_grab_value = get_value();
			if (!mousewheel_over_grabber) {
				grabbing_ratio = get_as_ratio();
				grabbing_from = grabber->get_transform().xform(mb->get_position()).x;
			}
			grab_focus();
			emit_signal("grabbed");
		} else {
			grabbing_grabber = false;
			mousewheel_over_grabber = false;
			emit_signal("ungrabbed");
			queue_redraw();
		}
	} else if (mb.is_valid() && mb->get_button_index() == MouseButton::RIGHT) {
		if (mb->is_pressed() && grabbing_grabber) {
			grabbing_grabber = false;
			mousewheel_over_grabber = false;
			set_value(pre_grab_value);
			emit_signal("ungrabbed");
		}
	}

	Ref<InputEventMouseMotion> mm = p_event;
	if (mm.is_valid() && grabbing_grabber) {
		if (mousewheel_over_grabber) {
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
				if (value_input_popup) {
					value_input_popup->hide();
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

	// Add a left margin to the stylebox to make the number align with the Label
	// when it's edited. The LineEdit "focus" stylebox uses the "normal" stylebox's
	// default margins.
	Ref<StyleBox> stylebox = theme_cache.normal_style->duplicate();
	// EditorSpinSliders with a label have more space on the left, so add an
	// higher margin to match the location where the text begins.
	// The margin values below were determined by empirical testing.
	if (is_layout_rtl()) {
		stylebox->set_content_margin(SIDE_RIGHT, (!get_label().is_empty() ? 23 : 16) * EDSCALE);
	} else {
		stylebox->set_content_margin(SIDE_LEFT, (!get_label().is_empty() ? 23 : 16) * EDSCALE);
	}

	value_input->add_theme_style_override(CoreStringName(normal), stylebox);
}

void EditorSpinSlider::_draw_spin_slider() {
	updown_offset = Point2(-1, -1);

	RID ci = get_canvas_item();
	bool rtl = is_layout_rtl();
	const Vector2 size = get_size();
	const bool draw_label = !label.is_empty();

	Ref<StyleBox> sb = read_only ? theme_cache.read_only_style : theme_cache.normal_style;
	if (!flat) {
		draw_style_box(sb, Rect2(Vector2(), size));
	}
	const Size2 style_min_size = sb->get_minimum_size();
	int sep = 4 * EDSCALE;

	String numstr = get_text_value();

	int vofs = (size.height - theme_cache.font->get_height(theme_cache.font_size)) / 2 + theme_cache.font->get_ascent(theme_cache.font_size);

	Color fc = read_only ? theme_cache.font_uneditable_color : theme_cache.font_color;

	int label_width = 0;
	if (draw_label) {
		label_width = theme_cache.font->get_string_size(label, HORIZONTAL_ALIGNMENT_LEFT, -1, theme_cache.font_size).width;
		sep += sb->get_offset().x; // Make the label have the same margin on both sides, looks better.

		if (flat) {
			if (rtl) {
				draw_style_box(theme_cache.label_bg_style, Rect2(Vector2(size.width - (sb->get_offset().x * 2 + label_width), 0), Vector2(sb->get_offset().x * 2 + label_width, size.height)));
			} else {
				draw_style_box(theme_cache.label_bg_style, Rect2(Vector2(), Vector2(sb->get_offset().x * 2 + label_width, size.height)));
			}
		}
	}

	if (has_focus()) {
		draw_style_box(theme_cache.focus_style, Rect2(Vector2(), size));
	}

	if (draw_label) {
		Color lc = read_only ? theme_cache.read_only_label_color : theme_cache.label_color;

		if (rtl) {
			draw_string(theme_cache.font, Vector2(Math::round(size.width - sb->get_offset().x - label_width), vofs), label, HORIZONTAL_ALIGNMENT_RIGHT, -1, theme_cache.font_size, lc * Color(1, 1, 1, 0.5));
		} else {
			draw_string(theme_cache.font, Vector2(Math::round(sb->get_offset().x), vofs), label, HORIZONTAL_ALIGNMENT_LEFT, -1, theme_cache.font_size, lc * Color(1, 1, 1, 0.5));
		}
	}

	int number_width = size.width - style_min_size.width - label_width - sep;
	int suffix_start = numstr.length();
	RID num_rid = TS->create_shaped_text();
	TS->shaped_text_add_string(num_rid, numstr + U"\u2009" + suffix, theme_cache.font->get_rids(), theme_cache.font_size, theme_cache.font->get_opentype_features());

	float text_start = rtl ? Math::round(sb->get_offset().x) : Math::round(sb->get_offset().x + label_width + sep);
	Vector2 text_ofs = rtl ? Vector2(text_start + (number_width - TS->shaped_text_get_width(num_rid)), vofs) : Vector2(text_start, vofs);
	int v_size = TS->shaped_text_get_glyph_count(num_rid);
	const Glyph *glyphs = TS->shaped_text_get_glyphs(num_rid);
	for (int i = 0; i < v_size; i++) {
		for (int j = 0; j < glyphs[i].repeat; j++) {
			if (text_ofs.x >= text_start && (text_ofs.x + glyphs[i].advance) <= (text_start + number_width)) {
				Color color = fc;
				if (glyphs[i].start >= suffix_start) {
					color.a *= 0.4;
				}
				if (glyphs[i].font_rid != RID()) {
					TS->font_draw_glyph(glyphs[i].font_rid, ci, glyphs[i].font_size, text_ofs + Vector2(glyphs[i].x_off, glyphs[i].y_off), glyphs[i].index, color);
				} else if (((glyphs[i].flags & TextServer::GRAPHEME_IS_VIRTUAL) != TextServer::GRAPHEME_IS_VIRTUAL) && ((glyphs[i].flags & TextServer::GRAPHEME_IS_EMBEDDED_OBJECT) != TextServer::GRAPHEME_IS_EMBEDDED_OBJECT)) {
					TS->draw_hex_code_box(ci, glyphs[i].font_size, text_ofs + Vector2(glyphs[i].x_off, glyphs[i].y_off), glyphs[i].index, color);
				}
			}
			text_ofs.x += glyphs[i].advance;
		}
	}
	TS->free_rid(num_rid);

	if (!hide_slider) {
		if (editing_integer) {
			int icon_sep = MAX(0, theme_cache.updown_v_separation);
			int icon_max_height = (size.height - icon_sep - style_min_size.height) / 2;
			const Size2 up_icon_ms = theme_cache.up_icon->get_size();
			const Size2 down_icon_ms = theme_cache.down_icon->get_size();
			int up_icon_height = MAX(up_icon_ms.height, icon_max_height);
			int down_icon_height = MAX(down_icon_ms.height, icon_max_height);
			Size2 up_icon_size = Size2(up_icon_ms.width, up_icon_height);
			Size2 down_icon_size = Size2(down_icon_ms.width, down_icon_height);
			int updown_min_width = MAX(up_icon_size.width, down_icon_size.width);
			int icon_x_pos;

			if (rtl) {
				icon_x_pos = sb->get_margin(SIDE_LEFT);
				updown_offset.x = icon_x_pos + updown_min_width;
			} else {
				icon_x_pos = size.width - updown_min_width - sb->get_margin(SIDE_RIGHT);
				updown_offset.x = icon_x_pos;
			}

			int half_sep = icon_sep / 2;
			updown_offset.y = size.height * 0.5;
			Point2 up_icon_pos = Point2(icon_x_pos, updown_offset.y - up_icon_size.height - half_sep);
			Point2 down_icon_pos = Point2(icon_x_pos, updown_offset.y + half_sep);
			bool up_icon_disabled = get_value() == get_max() && !is_greater_allowed();
			bool down_icon_disabled = get_value() == get_min() && !is_lesser_allowed();
			float up_icon_alpha = up_icon_disabled ? 0.4 : 0.8;
			float down_icon_alpha = down_icon_disabled ? 0.4 : 0.8;

			if (read_only) {
				up_icon_alpha = 0.4;
				down_icon_alpha = 0.4;
			} else {
				if (up_icon_hovered && !up_icon_disabled) {
					up_icon_alpha = 1.0;
					draw_style_box(updown_pressed ? theme_cache.updown_pressed_style : theme_cache.updown_hovered_style, Rect2(up_icon_pos, up_icon_size));
				} else if (down_icon_hovered && !down_icon_disabled) {
					down_icon_alpha = 1.0;
					draw_style_box(updown_pressed ? theme_cache.updown_pressed_style : theme_cache.updown_hovered_style, Rect2(down_icon_pos, down_icon_size));
				}
			}

			up_icon_pos.y += MAX(0, (up_icon_height - up_icon_ms.height) / 2);
			down_icon_pos.y += MAX(0, (down_icon_height - down_icon_ms.height) / 2);
			draw_texture(theme_cache.up_icon, up_icon_pos, Color(1, 1, 1, up_icon_alpha));
			draw_texture(theme_cache.down_icon, down_icon_pos, Color(1, 1, 1, down_icon_alpha));

			if (grabber->is_visible()) {
				grabber->hide();
			}
		} else {
			// Ensure grabber half size is always even so it stays pixel-aligned.
			const int grabber_half_size = Math::round(EDSCALE) * 2;
			const int grabber_size = grabber_half_size * 2;
			const int gofs = grabber_half_size * 0.5;
			const int width = size.width - MAX(style_min_size.width, grabber_size);
			const int svofs = Math::round((size.height + vofs) * 0.5) - gofs;
			Color c = fc;

			// Calculate begin offset and grabber position.
			int begin = MAX(sb->get_margin(SIDE_LEFT), grabber_half_size);
			int grabber_pos = (rtl ? (1 - get_as_ratio()) : get_as_ratio()) * width;

			// Draw the horizontal slider's background.
			c.a = 0.2;
			draw_rect(Rect2(begin, svofs, width, grabber_half_size), c);

			c.a = 0.45;
			// Draw the horizontal slider's filled part.
			if (rtl) {
				draw_rect(Rect2(grabber_pos + begin, svofs, width - grabber_pos, grabber_half_size), c);
			} else {
				draw_rect(Rect2(begin, svofs, grabber_pos, grabber_half_size), c);
			}

			bool display_grabber = !read_only && (grabbing_grabber || mouse_over_spin || mouse_over_grabber) && !grabbing_spinner && !(value_input_popup && value_input_popup->is_visible());
			if (grabber->is_visible() != display_grabber) {
				grabber->set_visible(display_grabber);
			}

			const Rect2 grabber_rect = Rect2(grabber_pos + begin - grabber_half_size, svofs - gofs, grabber_size, grabber_size);
			grabbing_spinner_mouse_pos = get_global_position() + grabber_rect.get_center();

			if (display_grabber) {
				grabber->set_texture(mouse_over_grabber ? theme_cache.grabber_highlight_icon : theme_cache.grabber_icon);

				grabber->reset_size();
				grabber->set_position(grabber_rect.get_center() - grabber->get_size() * 0.5);

				if (mousewheel_over_grabber) {
					Input::get_singleton()->warp_mouse(grabber->get_global_position() + grabber_rect.size);
				}

				grabber_range = width;
			} else {
				c.a = 0.9;
				draw_rect(grabber_rect, c);
			}
		}
	}
}

void EditorSpinSlider::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE: {
			grabbing_spinner_speed = EDITOR_GET("interface/inspector/float_drag_speed");
			_update_value_input_stylebox();
		} break;

		case NOTIFICATION_LAYOUT_DIRECTION_CHANGED:
		case NOTIFICATION_TRANSLATION_CHANGED:
		case NOTIFICATION_THEME_CHANGED: {
			_update_value_input_stylebox();
		} break;

		case NOTIFICATION_INTERNAL_PROCESS: {
			if (value_input_dirty) {
				value_input_dirty = false;
				value_input->set_text(get_text_value());
			}
			set_process_internal(false);
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

			if (!updown_pressed) {
				_update_button_hover_state(get_local_mouse_position());

				queue_redraw();
			}
		} break;

		case NOTIFICATION_MOUSE_EXIT: {
			mouse_over_spin = false;
			up_icon_hovered = false;
			down_icon_hovered = false;
			if (!updown_pressed) {
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
				grabbing_spinner_speed = EDITOR_GET("interface/inspector/float_drag_speed");
			}
		} break;
	}
}

LineEdit *EditorSpinSlider::get_line_edit() {
	_ensure_input_popup();
	return value_input;
}

Size2 EditorSpinSlider::get_minimum_size() const {
	Size2 ms = theme_cache.normal_style->get_minimum_size();

	// TODO: Calculate minimum width.
	if (editing_integer) {
		int updown_height = theme_cache.up_icon->get_height() + theme_cache.down_icon->get_height() + MAX(theme_cache.updown_v_separation, 0);

		ms.height += MAX(updown_height, theme_cache.font->get_height(theme_cache.font_size));
	} else {
		ms.height += theme_cache.font->get_height(theme_cache.font_size);
	}

	return ms;
}

void EditorSpinSlider::set_hide_slider(bool p_hide) {
	if (hide_slider == p_hide) {
		return;
	}

	hide_slider = p_hide;
	queue_redraw();
}

bool EditorSpinSlider::is_hiding_slider() const {
	return hide_slider;
}

void EditorSpinSlider::set_editing_integer(bool p_editing_integer) {
	if (p_editing_integer == editing_integer) {
		return;
	}

	editing_integer = p_editing_integer;
	queue_redraw();
}

bool EditorSpinSlider::is_editing_integer() const {
	return editing_integer;
}

void EditorSpinSlider::set_label(const String &p_label) {
	if (label == p_label) {
		return;
	}

	label = p_label;
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
	if (value_input_popup) {
		value_input_popup->hide();
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
		if (value_input_popup) {
			value_input_popup->hide();
		}
	} else {
		// Enter or Esc was pressed.
		grab_focus();
	}

	emit_signal("value_focus_exited");
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

void EditorSpinSlider::_focus_entered() {
	if (read_only) {
		return;
	}

	_ensure_input_popup();
	value_input->set_text(get_text_value());
	value_input_popup->set_size(get_size());
	value_input->set_focus_next(find_next_valid_focus()->get_path());
	value_input->set_focus_previous(find_prev_valid_focus()->get_path());
	callable_mp((CanvasItem *)value_input_popup, &CanvasItem::show).call_deferred();
	callable_mp((Control *)value_input, &Control::grab_focus).call_deferred();
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

	ClassDB::bind_method(D_METHOD("set_hide_slider", "hide_slider"), &EditorSpinSlider::set_hide_slider);
	ClassDB::bind_method(D_METHOD("is_hiding_slider"), &EditorSpinSlider::is_hiding_slider);

	ClassDB::bind_method(D_METHOD("set_editing_integer", "editing_integer"), &EditorSpinSlider::set_editing_integer);
	ClassDB::bind_method(D_METHOD("is_editing_integer"), &EditorSpinSlider::is_editing_integer);

	ADD_PROPERTY(PropertyInfo(Variant::STRING, "label"), "set_label", "get_label");
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "suffix"), "set_suffix", "get_suffix");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "read_only"), "set_read_only", "is_read_only");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "flat"), "set_flat", "is_flat");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "hide_slider"), "set_hide_slider", "is_hiding_slider");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "editing_integer"), "set_editing_integer", "is_editing_integer");

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

void EditorSpinSlider::_ensure_input_popup() {
	if (value_input_popup) {
		return;
	}

	value_input_popup = memnew(Control);
	value_input_popup->set_anchors_and_offsets_preset(PRESET_FULL_RECT);
	add_child(value_input_popup);

	value_input = memnew(LineEdit);
	value_input->set_emoji_menu_enabled(false);
	value_input->set_focus_mode(FOCUS_CLICK);
	value_input_popup->add_child(value_input);
	value_input->set_anchors_and_offsets_preset(PRESET_FULL_RECT);
	value_input_popup->connect(SceneStringName(hidden), callable_mp(this, &EditorSpinSlider::_value_input_closed));
	value_input->connect(SceneStringName(text_submitted), callable_mp(this, &EditorSpinSlider::_value_input_submitted));
	value_input->connect(SceneStringName(focus_exited), callable_mp(this, &EditorSpinSlider::_value_focus_exited));
	value_input->connect(SceneStringName(gui_input), callable_mp(this, &EditorSpinSlider::_value_input_gui_input));

	if (is_inside_tree()) {
		_update_value_input_stylebox();
	}
}

EditorSpinSlider::EditorSpinSlider() {
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
