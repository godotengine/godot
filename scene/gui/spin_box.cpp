/*************************************************************************/
/*  spin_box.cpp                                                         */
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

#include "spin_box.h"
#include "core/math/expression.h"
#include "core/os/input.h"

Size2 SpinBox::get_minimum_size() const {
	Size2 ms = line_edit->get_combined_minimum_size();
	ms.width += last_w;
	return ms;
}

void SpinBox::_value_changed(double) {
	String value = String::num(get_value(), Math::range_step_decimals(get_step()));
	if (prefix != "") {
		value = prefix + " " + value;
	}
	if (suffix != "") {
		value += " " + suffix;
	}
	line_edit->set_text(value);
}

void SpinBox::_text_entered(const String &p_string) {
	Ref<Expression> expr;
	expr.instance();
	// Ignore the prefix and suffix in the expression
	Error err = expr->parse(p_string.trim_prefix(prefix + " ").trim_suffix(" " + suffix));
	if (err != OK) {
		return;
	}

	Variant value = expr->execute(Array(), nullptr, false);
	if (value.get_type() != Variant::NIL) {
		set_value(value);
	}
	_value_changed(0);
}

LineEdit *SpinBox::get_line_edit() {
	return line_edit;
}

void SpinBox::_line_edit_input(const Ref<InputEvent> &p_event) {
}

void SpinBox::_range_click_timeout() {
	if (!drag.enabled && Input::get_singleton()->is_mouse_button_pressed(BUTTON_LEFT)) {
		bool up = get_local_mouse_position().y < (get_size().height / 2);
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

void SpinBox::_release_mouse() {
	if (drag.enabled) {
		drag.enabled = false;
		Input::get_singleton()->set_mouse_mode(Input::MOUSE_MODE_VISIBLE);
		warp_mouse(drag.capture_pos);
	}
}

void SpinBox::_gui_input(const Ref<InputEvent> &p_event) {
	if (!is_editable()) {
		return;
	}

	Ref<InputEventMouseButton> mb = p_event;

	if (mb.is_valid() && mb->is_pressed()) {
		bool up = mb->get_position().y < (get_size().height / 2);

		switch (mb->get_button_index()) {
			case BUTTON_LEFT: {
				line_edit->grab_focus();

				set_value(get_value() + (up ? get_step() : -get_step()));

				range_click_timer->set_wait_time(0.6);
				range_click_timer->set_one_shot(true);
				range_click_timer->start();

				drag.allowed = true;
				drag.capture_pos = mb->get_position();
			} break;
			case BUTTON_RIGHT: {
				line_edit->grab_focus();
				set_value((up ? get_max() : get_min()));
			} break;
			case BUTTON_WHEEL_UP: {
				if (line_edit->has_focus()) {
					set_value(get_value() + get_step() * mb->get_factor());
					accept_event();
				}
			} break;
			case BUTTON_WHEEL_DOWN: {
				if (line_edit->has_focus()) {
					set_value(get_value() - get_step() * mb->get_factor());
					accept_event();
				}
			} break;
		}
	}

	if (mb.is_valid() && !mb->is_pressed() && mb->get_button_index() == BUTTON_LEFT) {
		//set_default_cursor_shape(CURSOR_ARROW);
		range_click_timer->stop();
		_release_mouse();
		drag.allowed = false;
	}

	Ref<InputEventMouseMotion> mm = p_event;

	if (mm.is_valid() && mm->get_button_mask() & BUTTON_MASK_LEFT) {
		if (drag.enabled) {
			drag.diff_y += mm->get_relative().y;
			float diff_y = -0.01 * Math::pow(ABS(drag.diff_y), 1.8f) * SGN(drag.diff_y);
			set_value(CLAMP(drag.base_val + get_step() * diff_y, get_min(), get_max()));
		} else if (drag.allowed && drag.capture_pos.distance_to(mm->get_position()) > 2) {
			Input::get_singleton()->set_mouse_mode(Input::MOUSE_MODE_CAPTURED);
			drag.enabled = true;
			drag.base_val = get_value();
			drag.diff_y = 0;
		}
	}
}

void SpinBox::_line_edit_focus_exit() {
	// discontinue because the focus_exit was caused by right-click context menu
	if (line_edit->get_menu()->is_visible()) {
		return;
	}

	_text_entered(line_edit->get_text());
}

inline void SpinBox::_adjust_width_for_icon(const Ref<Texture> &icon) {
	int w = icon->get_width();
	if (w != last_w) {
		line_edit->set_margin(MARGIN_RIGHT, -w);
		last_w = w;
	}
}

void SpinBox::_notification(int p_what) {
	if (p_what == NOTIFICATION_DRAW) {
		Ref<Texture> updown = get_icon("updown");

		_adjust_width_for_icon(updown);

		RID ci = get_canvas_item();
		Size2i size = get_size();

		updown->draw(ci, Point2i(size.width - updown->get_width(), (size.height - updown->get_height()) / 2));

	} else if (p_what == NOTIFICATION_FOCUS_EXIT) {
		//_value_changed(0);
	} else if (p_what == NOTIFICATION_ENTER_TREE) {
		_adjust_width_for_icon(get_icon("updown"));
		_value_changed(0);
	} else if (p_what == NOTIFICATION_EXIT_TREE) {
		_release_mouse();
	} else if (p_what == NOTIFICATION_THEME_CHANGED) {
		call_deferred("minimum_size_changed");
		get_line_edit()->call_deferred("minimum_size_changed");
	}
}

void SpinBox::set_align(LineEdit::Align p_align) {
	line_edit->set_align(p_align);
}

LineEdit::Align SpinBox::get_align() const {
	return line_edit->get_align();
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

void SpinBox::apply() {
	_text_entered(line_edit->get_text());
}

void SpinBox::_bind_methods() {
	//ClassDB::bind_method(D_METHOD("_value_changed"),&SpinBox::_value_changed);
	ClassDB::bind_method(D_METHOD("_gui_input"), &SpinBox::_gui_input);
	ClassDB::bind_method(D_METHOD("_text_entered"), &SpinBox::_text_entered);
	ClassDB::bind_method(D_METHOD("set_align", "align"), &SpinBox::set_align);
	ClassDB::bind_method(D_METHOD("get_align"), &SpinBox::get_align);
	ClassDB::bind_method(D_METHOD("set_suffix", "suffix"), &SpinBox::set_suffix);
	ClassDB::bind_method(D_METHOD("get_suffix"), &SpinBox::get_suffix);
	ClassDB::bind_method(D_METHOD("set_prefix", "prefix"), &SpinBox::set_prefix);
	ClassDB::bind_method(D_METHOD("get_prefix"), &SpinBox::get_prefix);
	ClassDB::bind_method(D_METHOD("set_editable", "editable"), &SpinBox::set_editable);
	ClassDB::bind_method(D_METHOD("is_editable"), &SpinBox::is_editable);
	ClassDB::bind_method(D_METHOD("apply"), &SpinBox::apply);
	ClassDB::bind_method(D_METHOD("_line_edit_focus_exit"), &SpinBox::_line_edit_focus_exit);
	ClassDB::bind_method(D_METHOD("get_line_edit"), &SpinBox::get_line_edit);
	ClassDB::bind_method(D_METHOD("_line_edit_input"), &SpinBox::_line_edit_input);
	ClassDB::bind_method(D_METHOD("_range_click_timeout"), &SpinBox::_range_click_timeout);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "align", PROPERTY_HINT_ENUM, "Left,Center,Right,Fill"), "set_align", "get_align");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "editable"), "set_editable", "is_editable");
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "prefix"), "set_prefix", "get_prefix");
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "suffix"), "set_suffix", "get_suffix");
}

SpinBox::SpinBox() {
	last_w = 0;
	line_edit = memnew(LineEdit);
	add_child(line_edit);

	line_edit->set_anchors_and_margins_preset(Control::PRESET_WIDE);
	line_edit->set_mouse_filter(MOUSE_FILTER_PASS);
	//connect("value_changed",this,"_value_changed");
	line_edit->connect("text_entered", this, "_text_entered", Vector<Variant>(), CONNECT_DEFERRED);
	line_edit->connect("focus_exited", this, "_line_edit_focus_exit", Vector<Variant>(), CONNECT_DEFERRED);
	line_edit->connect("gui_input", this, "_line_edit_input");

	range_click_timer = memnew(Timer);
	range_click_timer->connect("timeout", this, "_range_click_timeout");
	add_child(range_click_timer);
}
