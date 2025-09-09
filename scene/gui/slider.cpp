/**************************************************************************/
/*  slider.cpp                                                            */
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

#include "slider.h"
#include "core/os/input.h"
#include "core/os/keyboard.h"

Size2 Slider::get_minimum_size() const {
	Ref<StyleBox> style = get_stylebox("slider");
	Size2i ss = style->get_minimum_size() + style->get_center_size();

	Ref<Texture> grabber = get_icon("grabber");
	Size2i rs = grabber->get_size();

	if (orientation == HORIZONTAL) {
		return Size2i(ss.width, MAX(ss.height, rs.height));
	} else {
		return Size2i(MAX(ss.width, rs.width), ss.height);
	}
}

void Slider::_gui_input(Ref<InputEvent> p_event) {
	ERR_FAIL_COND(p_event.is_null());

	if (!editable) {
		return;
	}

	Ref<InputEventMouseButton> mb = p_event;

	if (mb.is_valid()) {
		if (mb->get_button_index() == BUTTON_LEFT) {
			if (mb->is_pressed()) {
				Ref<Texture> grabber = get_icon(mouse_inside || has_focus() ? "grabber_highlight" : "grabber");
				grab.pos = orientation == VERTICAL ? mb->get_position().y : mb->get_position().x;

				double grab_width = (double)grabber->get_size().width;
				double grab_height = (double)grabber->get_size().height;
				double max = orientation == VERTICAL ? get_size().height - grab_height : get_size().width - grab_width;
				if (orientation == VERTICAL) {
					set_as_ratio(1 - (((double)grab.pos - (grab_height / 2.0)) / max));
				} else {
					set_as_ratio(((double)grab.pos - (grab_width / 2.0)) / max);
				}
				grab.active = true;
				grab.uvalue = get_as_ratio();

				emit_signal("drag_started");
			} else {
				grab.active = false;

				const bool value_changed = !Math::is_equal_approx((double)grab.uvalue, get_as_ratio());
				emit_signal("drag_ended", value_changed);
			}
		} else if (scrollable) {
			if (mb->is_pressed() && mb->get_button_index() == BUTTON_WHEEL_UP) {
				grab_focus();
				set_value(get_value() + get_step());
			} else if (mb->is_pressed() && mb->get_button_index() == BUTTON_WHEEL_DOWN) {
				grab_focus();
				set_value(get_value() - get_step());
			}
		}
	}

	Ref<InputEventMouseMotion> mm = p_event;

	if (mm.is_valid()) {
		if (grab.active) {
			Size2i size = get_size();
			Ref<Texture> grabber = get_icon("grabber");
			float motion = (orientation == VERTICAL ? mm->get_position().y : mm->get_position().x) - grab.pos;
			if (orientation == VERTICAL) {
				motion = -motion;
			}
			float areasize = orientation == VERTICAL ? size.height - grabber->get_size().height : size.width - grabber->get_size().width;
			if (areasize <= 0) {
				return;
			}
			float umotion = motion / float(areasize);
			set_as_ratio(grab.uvalue + umotion);
		}
	}

	Input *input = Input::get_singleton();
	Ref<InputEventJoypadMotion> joypadmotion_event = p_event;
	Ref<InputEventJoypadButton> joypadbutton_event = p_event;
	bool is_joypad_event = (joypadmotion_event.is_valid() || joypadbutton_event.is_valid());

	if (!mm.is_valid() && !mb.is_valid()) {
		if (p_event->is_action_pressed("ui_left", true)) {
			if (orientation != HORIZONTAL) {
				return;
			}
			if (is_joypad_event) {
				if (!input->is_action_just_pressed("ui_left")) {
					return;
				}
				set_process_internal(true);
			}
			set_value(get_value() - (custom_step >= 0 ? custom_step : get_step()));
			accept_event();
		} else if (p_event->is_action_pressed("ui_right", true)) {
			if (orientation != HORIZONTAL) {
				return;
			}
			if (is_joypad_event) {
				if (!input->is_action_just_pressed("ui_right")) {
					return;
				}
				set_process_internal(true);
			}
			set_value(get_value() + (custom_step >= 0 ? custom_step : get_step()));
			accept_event();
		} else if (p_event->is_action_pressed("ui_up", true)) {
			if (orientation != VERTICAL) {
				return;
			}
			if (is_joypad_event) {
				if (!input->is_action_just_pressed("ui_up")) {
					return;
				}
				set_process_internal(true);
			}
			set_value(get_value() + (custom_step >= 0 ? custom_step : get_step()));
			accept_event();
		} else if (p_event->is_action_pressed("ui_down", true)) {
			if (orientation != VERTICAL) {
				return;
			}
			if (is_joypad_event) {
				if (!input->is_action_just_pressed("ui_down")) {
					return;
				}
				set_process_internal(true);
			}
			set_value(get_value() - (custom_step >= 0 ? custom_step : get_step()));
			accept_event();
		} else if (p_event->is_action("ui_home") && p_event->is_pressed()) {
			set_value(get_min());
			accept_event();
		} else if (p_event->is_action("ui_end") && p_event->is_pressed()) {
			set_value(get_max());
			accept_event();
		}
	}
}

void Slider::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_INTERNAL_PROCESS: {
			Input *input = Input::get_singleton();

			if (input->is_action_just_released("ui_left") || input->is_action_just_released("ui_right") || input->is_action_just_released("ui_up") || input->is_action_just_released("ui_down")) {
				gamepad_event_delay_ms = DEFAULT_GAMEPAD_EVENT_DELAY_MS;
				set_process_internal(false);
				return;
			}

			gamepad_event_delay_ms -= get_process_delta_time();
			if (gamepad_event_delay_ms <= 0) {
				gamepad_event_delay_ms = GAMEPAD_EVENT_REPEAT_RATE_MS + gamepad_event_delay_ms;
				if (orientation == HORIZONTAL) {
					if (input->is_action_pressed("ui_left")) {
						set_value(get_value() - (custom_step >= 0 ? custom_step : get_step()));
					}

					if (input->is_action_pressed("ui_right")) {
						set_value(get_value() + (custom_step >= 0 ? custom_step : get_step()));
					}
				} else if (orientation == VERTICAL) {
					if (input->is_action_pressed("ui_down")) {
						set_value(get_value() - (custom_step >= 0 ? custom_step : get_step()));
					}

					if (input->is_action_pressed("ui_up")) {
						set_value(get_value() + (custom_step >= 0 ? custom_step : get_step()));
					}
				}
			}
		} break;
		case NOTIFICATION_THEME_CHANGED: {
			minimum_size_changed();
			update();
		} break;
		case NOTIFICATION_MOUSE_ENTER: {
			mouse_inside = true;
			update();
		} break;
		case NOTIFICATION_MOUSE_EXIT: {
			mouse_inside = false;
			update();
		} break;
		case NOTIFICATION_VISIBILITY_CHANGED: // fallthrough
		case NOTIFICATION_EXIT_TREE: {
			mouse_inside = false;
			grab.active = false;
		} break;
		case NOTIFICATION_DRAW: {
			RID ci = get_canvas_item();
			Size2i size = get_size();
			Ref<StyleBox> style = get_stylebox("slider");
			bool highlighted = mouse_inside || has_focus();
			Ref<StyleBox> grabber_area = get_stylebox(highlighted ? "grabber_area_highlight" : "grabber_area");
			Ref<Texture> grabber = get_icon(editable ? (highlighted ? "grabber_highlight" : "grabber") : "grabber_disabled");
			Ref<Texture> tick = get_icon("tick");
			double ratio = Math::is_nan(get_as_ratio()) ? 0 : get_as_ratio();

			if (orientation == VERTICAL) {
				int widget_width = style->get_minimum_size().width + style->get_center_size().width;
				float areasize = size.height - grabber->get_size().height;
				style->draw(ci, Rect2i(Point2i(size.width / 2 - widget_width / 2, 0), Size2i(widget_width, size.height)));
				grabber_area->draw(ci, Rect2i(Point2i((size.width - widget_width) / 2, size.height - areasize * ratio - grabber->get_size().height / 2), Size2i(widget_width, areasize * ratio + grabber->get_size().height / 2)));

				if (ticks > 1) {
					int grabber_offset = (grabber->get_size().height / 2 - tick->get_height() / 2);
					for (int i = 0; i < ticks; i++) {
						if (!ticks_on_borders && (i == 0 || i + 1 == ticks)) {
							continue;
						}
						int ofs = (i * areasize / (ticks - 1)) + grabber_offset;
						tick->draw(ci, Point2i((size.width - widget_width) / 2, ofs));
					}
				}
				grabber->draw(ci, Point2i(size.width / 2 - grabber->get_size().width / 2, size.height - ratio * areasize - grabber->get_size().height));
			} else {
				int widget_height = style->get_minimum_size().height + style->get_center_size().height;
				float areasize = size.width - grabber->get_size().width;

				style->draw(ci, Rect2i(Point2i(0, (size.height - widget_height) / 2), Size2i(size.width, widget_height)));
				grabber_area->draw(ci, Rect2i(Point2i(0, (size.height - widget_height) / 2), Size2i(areasize * ratio + grabber->get_size().width / 2, widget_height)));

				if (ticks > 1) {
					int grabber_offset = (grabber->get_size().width / 2 - tick->get_width() / 2);
					for (int i = 0; i < ticks; i++) {
						if ((!ticks_on_borders) && ((i == 0) || ((i + 1) == ticks))) {
							continue;
						}
						int ofs = (i * areasize / (ticks - 1)) + grabber_offset;
						tick->draw(ci, Point2i(ofs, (size.height - widget_height) / 2));
					}
				}
				grabber->draw(ci, Point2i(ratio * areasize, size.height / 2 - grabber->get_size().height / 2));
			}

		} break;
	}
}

void Slider::set_custom_step(float p_custom_step) {
	custom_step = p_custom_step;
}

float Slider::get_custom_step() const {
	return custom_step;
}

void Slider::set_ticks(int p_count) {
	ticks = p_count;
	update();
}

int Slider::get_ticks() const {
	return ticks;
}

bool Slider::get_ticks_on_borders() const {
	return ticks_on_borders;
}

void Slider::set_ticks_on_borders(bool _tob) {
	ticks_on_borders = _tob;
	update();
}

void Slider::set_editable(bool p_editable) {
	if (editable != p_editable) {
		grab.active = false;
	}

	editable = p_editable;
	update();
}

bool Slider::is_editable() const {
	return editable;
}

void Slider::set_scrollable(bool p_scrollable) {
	scrollable = p_scrollable;
}

bool Slider::is_scrollable() const {
	return scrollable;
}

void Slider::_bind_methods() {
	ClassDB::bind_method(D_METHOD("_gui_input"), &Slider::_gui_input);
	ClassDB::bind_method(D_METHOD("set_ticks", "count"), &Slider::set_ticks);
	ClassDB::bind_method(D_METHOD("get_ticks"), &Slider::get_ticks);

	ClassDB::bind_method(D_METHOD("get_ticks_on_borders"), &Slider::get_ticks_on_borders);
	ClassDB::bind_method(D_METHOD("set_ticks_on_borders", "ticks_on_border"), &Slider::set_ticks_on_borders);

	ClassDB::bind_method(D_METHOD("set_editable", "editable"), &Slider::set_editable);
	ClassDB::bind_method(D_METHOD("is_editable"), &Slider::is_editable);
	ClassDB::bind_method(D_METHOD("set_scrollable", "scrollable"), &Slider::set_scrollable);
	ClassDB::bind_method(D_METHOD("is_scrollable"), &Slider::is_scrollable);

	ADD_SIGNAL(MethodInfo("drag_started"));
	ADD_SIGNAL(MethodInfo("drag_ended", PropertyInfo(Variant::BOOL, "value_changed")));

	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "editable"), "set_editable", "is_editable");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "scrollable"), "set_scrollable", "is_scrollable");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "tick_count", PROPERTY_HINT_RANGE, "0,4096,1"), "set_ticks", "get_ticks");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "ticks_on_borders"), "set_ticks_on_borders", "get_ticks_on_borders");
}

Slider::Slider(Orientation p_orientation) {
	orientation = p_orientation;
	mouse_inside = false;
	grab.active = false;
	ticks = 0;
	ticks_on_borders = false;
	custom_step = -1;
	editable = true;
	scrollable = true;
	set_focus_mode(FOCUS_ALL);
}
