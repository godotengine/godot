/*************************************************************************/
/*  editor_spin_slider.cpp                                               */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2018 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2018 Godot Engine contributors (cf. AUTHORS.md)    */
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

#include "editor_spin_slider.h"
#include "editor_scale.h"
#include "os/input.h"
String EditorSpinSlider::get_text_value() const {
	int zeros = Math::step_decimals(get_step());
	return String::num(get_value(), zeros);
}
void EditorSpinSlider::_gui_input(const Ref<InputEvent> &p_event) {

	Ref<InputEventMouseButton> mb = p_event;
	if (mb.is_valid() && mb->get_button_index() == BUTTON_LEFT) {

		if (mb->is_pressed()) {

			if (updown_offset != -1 && mb->get_position().x > updown_offset) {
				//there is an updown, so use it.
				if (mb->get_position().y < get_size().height / 2) {
					set_value(get_value() + get_step());
				} else {
					set_value(get_value() - get_step());
				}
				return;
			} else {

				grabbing_spinner_attempt = true;
				grabbing_spinner = false;
				grabbing_spinner_mouse_pos = Input::get_singleton()->get_mouse_position();
			}
		} else {

			if (grabbing_spinner_attempt) {

				if (grabbing_spinner) {

					Input::get_singleton()->set_mouse_mode(Input::MOUSE_MODE_VISIBLE);
					Input::get_singleton()->warp_mouse_position(grabbing_spinner_mouse_pos);
					update();
				} else {
					Rect2 gr = get_global_rect();
					value_input->set_text(get_text_value());
					value_input->set_position(gr.position);
					value_input->set_size(gr.size);
					value_input->call_deferred("show_modal");
					value_input->call_deferred("grab_focus");
					value_input->call_deferred("select_all");
				}

				grabbing_spinner = false;
				grabbing_spinner_attempt = false;
			}
		}
	}

	Ref<InputEventMouseMotion> mm = p_event;
	if (mm.is_valid()) {

		if (grabbing_spinner_attempt) {

			if (!grabbing_spinner) {
				Input::get_singleton()->set_mouse_mode(Input::MOUSE_MODE_CAPTURED);
				grabbing_spinner = true;
			}

			double v = get_value();

			double diff_x = mm->get_relative().x;
			diff_x = Math::pow(ABS(diff_x), 1.8) * SGN(diff_x);
			diff_x *= 0.1;

			v += diff_x * get_step();

			set_value(v);

		} else if (updown_offset != -1) {
			bool new_hover = (mm->get_position().x > updown_offset);
			if (new_hover != hover_updown) {
				hover_updown = new_hover;
				update();
			}
		}
	}

	Ref<InputEventKey> k = p_event;
	if (k.is_valid() && k->is_pressed() && k->is_action("ui_accept")) {
		Rect2 gr = get_global_rect();
		value_input->set_text(get_text_value());
		value_input->set_position(gr.position);
		value_input->set_size(gr.size);
		value_input->call_deferred("show_modal");
		value_input->call_deferred("grab_focus");
		value_input->call_deferred("select_all");
	}
}

void EditorSpinSlider::_value_input_closed() {
	set_value(value_input->get_text().to_double());
}

void EditorSpinSlider::_value_input_entered(const String &p_text) {
	set_value(p_text.to_double());
	value_input->hide();
}

void EditorSpinSlider::_grabber_gui_input(const Ref<InputEvent> &p_event) {

	Ref<InputEventMouseButton> mb = p_event;
	if (mb.is_valid() && mb->get_button_index() == BUTTON_LEFT) {

		if (mb->is_pressed()) {

			grabbing_grabber = true;
			grabbing_ratio = get_as_ratio();
			grabbing_from = grabber->get_transform().xform(mb->get_position()).x;
		} else {
			grabbing_grabber = false;
		}
	}

	Ref<InputEventMouseMotion> mm = p_event;
	if (mm.is_valid() && grabbing_grabber) {

		float grabbing_ofs = (grabber->get_transform().xform(mm->get_position()).x - grabbing_from) / float(grabber_range);
		set_as_ratio(grabbing_ratio + grabbing_ofs);
		update();
	}
}

void EditorSpinSlider::_notification(int p_what) {

	if (p_what == MainLoop::NOTIFICATION_WM_FOCUS_OUT || p_what == MainLoop::NOTIFICATION_WM_FOCUS_OUT) {
		if (grabbing_spinner) {
			Input::get_singleton()->set_mouse_mode(Input::MOUSE_MODE_VISIBLE);
			grabbing_spinner = false;
			grabbing_spinner_attempt = false;
		}
	}

	if (p_what == NOTIFICATION_DRAW) {

		updown_offset = -1;

		Ref<StyleBox> sb = get_stylebox("normal", "LineEdit");
		draw_style_box(sb, Rect2(Vector2(), get_size()));
		Ref<Font> font = get_font("font", "LineEdit");

		int avail_width = get_size().width - sb->get_minimum_size().width - sb->get_minimum_size().width;
		avail_width -= font->get_string_size(label).width;
		Ref<Texture> updown = get_icon("updown", "SpinBox");

		if (get_step() == 1) {
			avail_width -= updown->get_width();
		}

		if (has_focus()) {
			Ref<StyleBox> focus = get_stylebox("focus", "LineEdit");
			draw_style_box(focus, Rect2(Vector2(), get_size()));
		}

		String numstr = get_text_value();

		int vofs = (get_size().height - font->get_height()) / 2 + font->get_ascent();

		Color fc = get_color("font_color", "LineEdit");

		int label_ofs = sb->get_offset().x + avail_width;
		draw_string(font, Vector2(label_ofs, vofs), label, fc * Color(1, 1, 1, 0.5));
		draw_string(font, Vector2(sb->get_offset().x, vofs), numstr, fc, avail_width);

		if (get_step() == 1) {
			Ref<Texture> updown = get_icon("updown", "SpinBox");
			int updown_vofs = (get_size().height - updown->get_height()) / 2;
			updown_offset = get_size().width - sb->get_margin(MARGIN_RIGHT) - updown->get_width();
			Color c(1, 1, 1);
			if (hover_updown) {
				c *= Color(1.2, 1.2, 1.2);
			}
			draw_texture(updown, Vector2(updown_offset, updown_vofs), c);
			if (grabber->is_visible()) {
				grabber->hide();
			}
		} else if (!hide_slider) {
			int grabber_w = 4 * EDSCALE;
			int width = get_size().width - sb->get_minimum_size().width - grabber_w;
			int ofs = sb->get_offset().x;
			int svofs = (get_size().height + vofs) / 2 - 1;
			Color c = fc;
			c.a = 0.2;

			draw_rect(Rect2(ofs, svofs + 1, width, 2 * EDSCALE), c);
			int gofs = get_as_ratio() * width;
			c.a = 0.9;
			Rect2 grabber_rect = Rect2(ofs + gofs, svofs + 1, grabber_w, 2 * EDSCALE);
			draw_rect(grabber_rect, c);

			bool display_grabber = (mouse_over_spin || mouse_over_grabber) && !grabbing_spinner;
			if (grabber->is_visible() != display_grabber) {
				if (display_grabber) {
					grabber->show();
				} else {
					grabber->hide();
				}
			}

			if (display_grabber) {
				Ref<Texture> grabber_tex;
				if (mouse_over_grabber) {
					grabber_tex = get_icon("grabber_highlight", "HSlider");
				} else {
					grabber_tex = get_icon("grabber", "HSlider");
				}

				if (grabber->get_texture() != grabber_tex) {
					grabber->set_texture(grabber_tex);
				}

				grabber->set_size(Size2(0, 0));
				grabber->set_position(get_global_position() + grabber_rect.position + grabber_rect.size * 0.5 - grabber->get_size() * 0.5);
				grabber_range = width;
			}
		}
	}

	if (p_what == NOTIFICATION_MOUSE_ENTER) {

		mouse_over_spin = true;
		update();
	}
	if (p_what == NOTIFICATION_MOUSE_EXIT) {

		mouse_over_spin = false;
		update();
	}
}

Size2 EditorSpinSlider::get_minimum_size() const {

	Ref<StyleBox> sb = get_stylebox("normal", "LineEdit");
	Ref<Font> font = get_font("font", "LineEdit");

	Size2 ms = sb->get_minimum_size();
	ms.height += font->get_height();

	return ms;
}

void EditorSpinSlider::set_hide_slider(bool p_hide) {
	hide_slider = p_hide;
	update();
}

bool EditorSpinSlider::is_hiding_slider() const {
	return hide_slider;
}

void EditorSpinSlider::set_label(const String &p_label) {
	label = p_label;
	update();
}

String EditorSpinSlider::get_label() const {
	return label;
}

void EditorSpinSlider::_grabber_mouse_entered() {
	mouse_over_grabber = true;
	update();
}

void EditorSpinSlider::_grabber_mouse_exited() {
	mouse_over_grabber = false;
	update();
}

void EditorSpinSlider::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_label", "label"), &EditorSpinSlider::set_label);
	ClassDB::bind_method(D_METHOD("get_label"), &EditorSpinSlider::get_label);

	ClassDB::bind_method(D_METHOD("_gui_input"), &EditorSpinSlider::_gui_input);
	ClassDB::bind_method(D_METHOD("_grabber_mouse_entered"), &EditorSpinSlider::_grabber_mouse_entered);
	ClassDB::bind_method(D_METHOD("_grabber_mouse_exited"), &EditorSpinSlider::_grabber_mouse_exited);
	ClassDB::bind_method(D_METHOD("_grabber_gui_input"), &EditorSpinSlider::_grabber_gui_input);
	ClassDB::bind_method(D_METHOD("_value_input_closed"), &EditorSpinSlider::_value_input_closed);
	ClassDB::bind_method(D_METHOD("_value_input_entered"), &EditorSpinSlider::_value_input_entered);

	ADD_PROPERTY(PropertyInfo(Variant::STRING, "label"), "set_label", "get_label");
}

EditorSpinSlider::EditorSpinSlider() {

	grabbing_spinner_attempt = false;
	grabbing_spinner = false;

	set_focus_mode(FOCUS_ALL);
	updown_offset = -1;
	hover_updown = false;
	grabber = memnew(TextureRect);
	add_child(grabber);
	grabber->hide();
	grabber->set_as_toplevel(true);
	grabber->set_mouse_filter(MOUSE_FILTER_STOP);
	grabber->connect("mouse_entered", this, "_grabber_mouse_entered");
	grabber->connect("mouse_exited", this, "_grabber_mouse_exited");
	grabber->connect("gui_input", this, "_grabber_gui_input");
	mouse_over_spin = false;
	mouse_over_grabber = false;
	grabbing_grabber = false;
	grabber_range = 1;
	value_input = memnew(LineEdit);
	add_child(value_input);
	value_input->set_as_toplevel(true);
	value_input->hide();
	value_input->connect("modal_closed", this, "_value_input_closed");
	value_input->connect("text_entered", this, "_value_input_entered");
	hide_slider = false;
}
