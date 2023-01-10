/**************************************************************************/
/*  android_input_handler.cpp                                             */
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

#include "android_input_handler.h"

#include "android_keys_utils.h"

#include "core/os/os.h"

void AndroidInputHandler::process_joy_event(const JoypadEvent &p_event) {
	switch (p_event.type) {
		case JOY_EVENT_BUTTON:
			input->joy_button(p_event.device, p_event.index, p_event.pressed);
			break;
		case JOY_EVENT_AXIS:
			input->joy_axis(p_event.device, p_event.index, p_event.value);
			break;
		case JOY_EVENT_HAT:
			input->joy_hat(p_event.device, p_event.hat);
			break;
		default:
			return;
	}
}

void AndroidInputHandler::_set_key_modifier_state(Ref<InputEventWithModifiers> ev) const {
	ev->set_shift(shift_mem);
	ev->set_alt(alt_mem);
	ev->set_metakey(meta_mem);
	ev->set_control(control_mem);
}

void AndroidInputHandler::process_key_event(int p_scancode, int p_physical_scancode, int p_unicode, bool p_pressed) {
	Ref<InputEventKey> ev;
	ev.instance();

	unsigned int physical_scancode = godot_code_from_android_code(p_physical_scancode);
	unsigned int scancode = physical_scancode;
	if (p_scancode != 0) {
		scancode = godot_code_from_unicode(p_scancode);
	}

	switch (physical_scancode) {
		case KEY_SHIFT: {
			shift_mem = p_pressed;
		} break;
		case KEY_ALT: {
			alt_mem = p_pressed;
		} break;
		case KEY_CONTROL: {
			control_mem = p_pressed;
		} break;
		case KEY_META: {
			meta_mem = p_pressed;
		} break;
		default:
			break;
	}

	ev->set_scancode(scancode);
	ev->set_physical_scancode(physical_scancode);
	ev->set_unicode(p_unicode);
	ev->set_pressed(p_pressed);

	_set_key_modifier_state(ev);

	if (p_physical_scancode == AKEYCODE_BACK) {
		if (MainLoop *main_loop = OS::get_singleton()->get_main_loop()) {
			main_loop->call_deferred("notification", MainLoop::NOTIFICATION_WM_GO_BACK_REQUEST);
		}
	}

	input->parse_input_event(ev);
}

void AndroidInputHandler::_parse_all_touch(bool p_pressed, bool p_double_tap) {
	if (touch.size()) {
		//end all if exist
		for (int i = 0; i < touch.size(); i++) {
			Ref<InputEventScreenTouch> ev;
			ev.instance();
			ev->set_index(touch[i].id);
			ev->set_pressed(p_pressed);
			ev->set_position(touch[i].pos);
			ev->set_double_tap(p_double_tap);
			input->parse_input_event(ev);
		}
	}
}

void AndroidInputHandler::_release_all_touch() {
	_parse_all_touch(false, false);
	touch.clear();
}

void AndroidInputHandler::process_touch_event(int p_event, int p_pointer, const Vector<TouchPos> &p_points, bool p_double_tap) {
	switch (p_event) {
		case AMOTION_EVENT_ACTION_DOWN: { //gesture begin
			// Release any remaining touches or mouse event
			_release_mouse_event_info();
			_release_all_touch();

			touch.resize(p_points.size());
			for (int i = 0; i < p_points.size(); i++) {
				touch.write[i].id = p_points[i].id;
				touch.write[i].pos = p_points[i].pos;
			}

			//send touch
			_parse_all_touch(true, p_double_tap);

		} break;
		case AMOTION_EVENT_ACTION_MOVE: { //motion
			if (touch.size() != p_points.size()) {
				return;
			}

			for (int i = 0; i < touch.size(); i++) {
				int idx = -1;
				for (int j = 0; j < p_points.size(); j++) {
					if (touch[i].id == p_points[j].id) {
						idx = j;
						break;
					}
				}

				ERR_CONTINUE(idx == -1);

				if (touch[i].pos == p_points[idx].pos) {
					continue; //no move unncesearily
				}

				Ref<InputEventScreenDrag> ev;
				ev.instance();
				ev->set_index(touch[i].id);
				ev->set_position(p_points[idx].pos);
				ev->set_relative(p_points[idx].pos - touch[i].pos);
				input->parse_input_event(ev);
				touch.write[i].pos = p_points[idx].pos;
			}

		} break;
		case AMOTION_EVENT_ACTION_CANCEL:
		case AMOTION_EVENT_ACTION_UP: { //release
			_release_all_touch();
		} break;
		case AMOTION_EVENT_ACTION_POINTER_DOWN: { // add touch
			for (int i = 0; i < p_points.size(); i++) {
				if (p_points[i].id == p_pointer) {
					TouchPos tp = p_points[i];
					touch.push_back(tp);

					Ref<InputEventScreenTouch> ev;
					ev.instance();

					ev->set_index(tp.id);
					ev->set_pressed(true);
					ev->set_position(tp.pos);
					input->parse_input_event(ev);

					break;
				}
			}
		} break;
		case AMOTION_EVENT_ACTION_POINTER_UP: { // remove touch
			for (int i = 0; i < touch.size(); i++) {
				if (touch[i].id == p_pointer) {
					Ref<InputEventScreenTouch> ev;
					ev.instance();
					ev->set_index(touch[i].id);
					ev->set_pressed(false);
					ev->set_position(touch[i].pos);
					input->parse_input_event(ev);
					touch.remove(i);

					break;
				}
			}
		} break;
	}
}

void AndroidInputHandler::_parse_mouse_event_info(int buttons_mask, bool p_pressed, bool p_double_click, bool p_source_mouse_relative) {
	if (!mouse_event_info.valid) {
		return;
	}

	Ref<InputEventMouseButton> ev;
	ev.instance();
	_set_key_modifier_state(ev);
	if (p_source_mouse_relative) {
		ev->set_position(hover_prev_pos);
		ev->set_global_position(hover_prev_pos);
	} else {
		ev->set_position(mouse_event_info.pos);
		ev->set_global_position(mouse_event_info.pos);
		hover_prev_pos = mouse_event_info.pos;
	}
	ev->set_pressed(p_pressed);
	int changed_button_mask = buttons_state ^ buttons_mask;

	buttons_state = buttons_mask;

	ev->set_button_index(_button_index_from_mask(changed_button_mask));
	ev->set_button_mask(buttons_mask);
	ev->set_doubleclick(p_double_click);
	input->parse_input_event(ev);
}

void AndroidInputHandler::_release_mouse_event_info(bool p_source_mouse_relative) {
	_parse_mouse_event_info(0, false, false, p_source_mouse_relative);
	mouse_event_info.valid = false;
}

void AndroidInputHandler::process_mouse_event(int p_event_action, int p_event_android_buttons_mask, Point2 p_event_pos, Vector2 p_delta, bool p_double_click, bool p_source_mouse_relative) {
	int event_buttons_mask = _android_button_mask_to_godot_button_mask(p_event_android_buttons_mask);
	switch (p_event_action) {
		case AMOTION_EVENT_ACTION_HOVER_MOVE: // hover move
		case AMOTION_EVENT_ACTION_HOVER_ENTER: // hover enter
		case AMOTION_EVENT_ACTION_HOVER_EXIT: { // hover exit
			// https://developer.android.com/reference/android/view/MotionEvent.html#ACTION_HOVER_ENTER
			Ref<InputEventMouseMotion> ev;
			ev.instance();
			_set_key_modifier_state(ev);
			ev->set_position(p_event_pos);
			ev->set_global_position(p_event_pos);
			ev->set_relative(p_event_pos - hover_prev_pos);
			input->parse_input_event(ev);
			hover_prev_pos = p_event_pos;
		} break;

		case AMOTION_EVENT_ACTION_DOWN:
		case AMOTION_EVENT_ACTION_BUTTON_PRESS: {
			// Release any remaining touches or mouse event
			_release_mouse_event_info();
			_release_all_touch();

			mouse_event_info.valid = true;
			mouse_event_info.pos = p_event_pos;
			_parse_mouse_event_info(event_buttons_mask, true, p_double_click, p_source_mouse_relative);
		} break;

		case AMOTION_EVENT_ACTION_UP:
		case AMOTION_EVENT_ACTION_CANCEL:
		case AMOTION_EVENT_ACTION_BUTTON_RELEASE: {
			_release_mouse_event_info(p_source_mouse_relative);
		} break;

		case AMOTION_EVENT_ACTION_MOVE: {
			if (!mouse_event_info.valid) {
				return;
			}

			Ref<InputEventMouseMotion> ev;
			ev.instance();
			_set_key_modifier_state(ev);
			if (p_source_mouse_relative) {
				ev->set_position(hover_prev_pos);
				ev->set_global_position(hover_prev_pos);
				ev->set_relative(p_event_pos);
			} else {
				ev->set_position(p_event_pos);
				ev->set_global_position(p_event_pos);
				ev->set_relative(p_event_pos - hover_prev_pos);
				mouse_event_info.pos = p_event_pos;
				hover_prev_pos = p_event_pos;
			}
			ev->set_button_mask(event_buttons_mask);
			input->parse_input_event(ev);
		} break;

		case AMOTION_EVENT_ACTION_SCROLL: {
			Ref<InputEventMouseButton> ev;
			ev.instance();
			_set_key_modifier_state(ev);
			if (p_source_mouse_relative) {
				ev->set_position(hover_prev_pos);
				ev->set_global_position(hover_prev_pos);
			} else {
				ev->set_position(p_event_pos);
				ev->set_global_position(p_event_pos);
			}
			ev->set_pressed(true);
			buttons_state = event_buttons_mask;
			if (p_delta.y > 0) {
				_wheel_button_click(event_buttons_mask, ev, BUTTON_WHEEL_UP, p_delta.y);
			} else if (p_delta.y < 0) {
				_wheel_button_click(event_buttons_mask, ev, BUTTON_WHEEL_DOWN, -p_delta.y);
			}

			if (p_delta.x > 0) {
				_wheel_button_click(event_buttons_mask, ev, BUTTON_WHEEL_RIGHT, p_delta.x);
			} else if (p_delta.x < 0) {
				_wheel_button_click(event_buttons_mask, ev, BUTTON_WHEEL_LEFT, -p_delta.x);
			}
		} break;
	}
}

void AndroidInputHandler::_wheel_button_click(int event_buttons_mask, const Ref<InputEventMouseButton> &ev, int wheel_button, float factor) {
	Ref<InputEventMouseButton> evd = ev->duplicate();
	evd->set_button_index(wheel_button);
	evd->set_button_mask(event_buttons_mask ^ (1 << (wheel_button - 1)));
	evd->set_factor(factor);
	input->parse_input_event(evd);
	Ref<InputEventMouseButton> evdd = evd->duplicate();
	evdd->set_pressed(false);
	evdd->set_button_mask(event_buttons_mask);
	input->parse_input_event(evdd);
}

void AndroidInputHandler::process_magnify(Point2 p_pos, float p_factor) {
	Ref<InputEventMagnifyGesture> magnify_event;
	magnify_event.instance();
	_set_key_modifier_state(magnify_event);
	magnify_event->set_position(p_pos);
	magnify_event->set_factor(p_factor);
	input->parse_input_event(magnify_event);
}

void AndroidInputHandler::process_pan(Point2 p_pos, Vector2 p_delta) {
	Ref<InputEventPanGesture> pan_event;
	pan_event.instance();
	_set_key_modifier_state(pan_event);
	pan_event->set_position(p_pos);
	pan_event->set_delta(p_delta);
	input->parse_input_event(pan_event);
}

int AndroidInputHandler::_button_index_from_mask(int button_mask) {
	switch (button_mask) {
		case BUTTON_MASK_LEFT:
			return BUTTON_LEFT;
		case BUTTON_MASK_RIGHT:
			return BUTTON_RIGHT;
		case BUTTON_MASK_MIDDLE:
			return BUTTON_MIDDLE;
		case BUTTON_MASK_XBUTTON1:
			return BUTTON_XBUTTON1;
		case BUTTON_MASK_XBUTTON2:
			return BUTTON_XBUTTON2;
		default:
			return 0;
	}
}

int AndroidInputHandler::_android_button_mask_to_godot_button_mask(int android_button_mask) {
	int godot_button_mask = 0;
	if (android_button_mask & AMOTION_EVENT_BUTTON_PRIMARY) {
		godot_button_mask |= BUTTON_MASK_LEFT;
	}
	if (android_button_mask & AMOTION_EVENT_BUTTON_SECONDARY) {
		godot_button_mask |= BUTTON_MASK_RIGHT;
	}
	if (android_button_mask & AMOTION_EVENT_BUTTON_TERTIARY) {
		godot_button_mask |= BUTTON_MASK_MIDDLE;
	}
	if (android_button_mask & AMOTION_EVENT_BUTTON_BACK) {
		godot_button_mask |= BUTTON_MASK_XBUTTON1;
	}
	if (android_button_mask & AMOTION_EVENT_BUTTON_FORWARD) {
		godot_button_mask |= BUTTON_MASK_XBUTTON2;
	}

	return godot_button_mask;
}

void AndroidInputHandler::joy_connection_changed(int p_device, bool p_connected, String p_name) {
	input->joy_connection_changed(p_device, p_connected, p_name, "");
}
