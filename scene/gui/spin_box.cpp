/*************************************************************************/
/*  spin_box.cpp                                                         */
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
#include "spin_box.h"
#include "os/input.h"

Size2 SpinBox::get_minimum_size() const {

	Size2 ms = line_edit->get_combined_minimum_size();
	ms.width += last_w;
	return ms;
}

void SpinBox::_value_changed(double) {

	String value = String::num(get_value(), Math::step_decimals(get_step()));
	if (prefix != "")
		value = prefix + " " + value;
	if (suffix != "")
		value += " " + suffix;
	line_edit->set_text(value);
}

void SpinBox::_text_entered(const String &p_string) {

	/*
	if (!p_string.is_numeric())
		return;
	*/
	String value = p_string;
	if (prefix != "" && p_string.begins_with(prefix))
		value = p_string.substr(prefix.length(), p_string.length() - prefix.length());
	set_value(value.to_double());
	_value_changed(0);
}

LineEdit *SpinBox::get_line_edit() {

	return line_edit;
}

void SpinBox::_line_edit_input(const InputEvent &p_event) {
}

void SpinBox::_range_click_timeout() {

	if (!drag.enabled && Input::get_singleton()->is_mouse_button_pressed(BUTTON_LEFT)) {

		bool up = get_local_mouse_pos().y < (get_size().height / 2);
		set_value(get_value() + (up ? get_step() : -get_step()));

		if (range_click_timer->is_one_shot()) {
			range_click_timer->set_wait_time(0.075);
			range_click_timer->set_one_shot(false);
			range_click_timer->start();
		}

	} else {
		range_click_timer->stop();
	}
}

void SpinBox::_gui_input(const InputEvent &p_event) {

	if (!is_editable()) {
		return;
	}
	if (p_event.type == InputEvent::MOUSE_BUTTON && p_event.mouse_button.pressed) {
		const InputEventMouseButton &mb = p_event.mouse_button;

		bool up = mb.y < (get_size().height / 2);

		switch (mb.button_index) {

			case BUTTON_LEFT: {

				set_value(get_value() + (up ? get_step() : -get_step()));

				range_click_timer->set_wait_time(0.6);
				range_click_timer->set_one_shot(true);
				range_click_timer->start();

				line_edit->grab_focus();
			} break;
			case BUTTON_RIGHT: {

				set_value((up ? get_max() : get_min()));
				line_edit->grab_focus();
			} break;
			case BUTTON_WHEEL_UP: {
				if (line_edit->has_focus()) {
					set_value(get_value() + get_step());
					accept_event();
				}
			} break;
			case BUTTON_WHEEL_DOWN: {
				if (line_edit->has_focus()) {
					set_value(get_value() - get_step());
					accept_event();
				}
			} break;
		}
	}

	if (p_event.type == InputEvent::MOUSE_BUTTON && p_event.mouse_button.pressed && p_event.mouse_button.button_index == 1) {

		//set_default_cursor_shape(CURSOR_VSIZE);
		Vector2 cpos = Vector2(p_event.mouse_button.x, p_event.mouse_button.y);
		drag.mouse_pos = cpos;
	}

	if (p_event.type == InputEvent::MOUSE_BUTTON && !p_event.mouse_button.pressed && p_event.mouse_button.button_index == 1) {

		//set_default_cursor_shape(CURSOR_ARROW);
		range_click_timer->stop();

		if (drag.enabled) {
			drag.enabled = false;
			Input::get_singleton()->set_mouse_mode(Input::MOUSE_MODE_VISIBLE);
			warp_mouse(drag.capture_pos);
		}
	}

	if (p_event.type == InputEvent::MOUSE_MOTION && p_event.mouse_button.button_mask & 1) {

		Vector2 cpos = Vector2(p_event.mouse_motion.x, p_event.mouse_motion.y);
		if (drag.enabled) {

			float diff_y = drag.mouse_pos.y - cpos.y;
			diff_y = Math::pow(ABS(diff_y), 1.8f) * SGN(diff_y);
			diff_y *= 0.1;

			drag.mouse_pos = cpos;
			drag.base_val = CLAMP(drag.base_val + get_step() * diff_y, get_min(), get_max());

			set_value(drag.base_val);

		} else if (drag.mouse_pos.distance_to(cpos) > 2) {

			Input::get_singleton()->set_mouse_mode(Input::MOUSE_MODE_CAPTURED);
			drag.enabled = true;
			drag.base_val = get_value();
			drag.mouse_pos = cpos;
			drag.capture_pos = cpos;
		}
	}
}

void SpinBox::_line_edit_focus_exit() {

	_text_entered(line_edit->get_text());
}

void SpinBox::_notification(int p_what) {

	if (p_what == NOTIFICATION_DRAW) {

		Ref<Texture> updown = get_icon("updown");

		int w = updown->get_width();
		if (w != last_w) {
			line_edit->set_margin(MARGIN_RIGHT, w);
			last_w = w;
		}

		RID ci = get_canvas_item();
		Size2i size = get_size();

		updown->draw(ci, Point2i(size.width - updown->get_width(), (size.height - updown->get_height()) / 2));

	} else if (p_what == NOTIFICATION_FOCUS_EXIT) {

		//_value_changed(0);
	} else if (p_what == NOTIFICATION_ENTER_TREE) {

		_value_changed(0);
	}
}

void SpinBox::set_suffix(const String &p_suffix) {

	suffix = p_suffix;
	_value_changed(0);
}

String SpinBox::get_suffix() const {

	return suffix;
}

void SpinBox::set_prefix(const String &p_prefix) {

	prefix = p_prefix;
	_value_changed(0);
}

String SpinBox::get_prefix() const {

	return prefix;
}

void SpinBox::set_editable(bool p_editable) {
	line_edit->set_editable(p_editable);
}

bool SpinBox::is_editable() const {

	return line_edit->is_editable();
}

void SpinBox::_bind_methods() {

	//ClassDB::bind_method(D_METHOD("_value_changed"),&SpinBox::_value_changed);
	ClassDB::bind_method(D_METHOD("_gui_input"), &SpinBox::_gui_input);
	ClassDB::bind_method(D_METHOD("_text_entered"), &SpinBox::_text_entered);
	ClassDB::bind_method(D_METHOD("set_suffix", "suffix"), &SpinBox::set_suffix);
	ClassDB::bind_method(D_METHOD("get_suffix"), &SpinBox::get_suffix);
	ClassDB::bind_method(D_METHOD("set_prefix", "prefix"), &SpinBox::set_prefix);
	ClassDB::bind_method(D_METHOD("get_prefix"), &SpinBox::get_prefix);
	ClassDB::bind_method(D_METHOD("set_editable", "editable"), &SpinBox::set_editable);
	ClassDB::bind_method(D_METHOD("is_editable"), &SpinBox::is_editable);
	ClassDB::bind_method(D_METHOD("_line_edit_focus_exit"), &SpinBox::_line_edit_focus_exit);
	ClassDB::bind_method(D_METHOD("get_line_edit"), &SpinBox::get_line_edit);
	ClassDB::bind_method(D_METHOD("_line_edit_input"), &SpinBox::_line_edit_input);
	ClassDB::bind_method(D_METHOD("_range_click_timeout"), &SpinBox::_range_click_timeout);

	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "editable"), "set_editable", "is_editable");
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "prefix"), "set_prefix", "get_prefix");
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "suffix"), "set_suffix", "get_suffix");
}

SpinBox::SpinBox() {

	last_w = 0;
	line_edit = memnew(LineEdit);
	add_child(line_edit);

	line_edit->set_area_as_parent_rect();
	//connect("value_changed",this,"_value_changed");
	line_edit->connect("text_entered", this, "_text_entered", Vector<Variant>(), CONNECT_DEFERRED);
	line_edit->connect("focus_exited", this, "_line_edit_focus_exit", Vector<Variant>(), CONNECT_DEFERRED);
	line_edit->connect("gui_input", this, "_line_edit_input");
	drag.enabled = false;

	range_click_timer = memnew(Timer);
	range_click_timer->connect("timeout", this, "_range_click_timeout");
	add_child(range_click_timer);
}
