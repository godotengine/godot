/*************************************************************************/
/*  android_input_handler.cpp                                            */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2021 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2021 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "android_input_handler.h"

#include "android_keys_utils.h"
#include "display_server_android.h"

void AndroidInputHandler::process_joy_event(AndroidInputHandler::JoypadEvent p_event) {
	switch (p_event.type) {
		case JOY_EVENT_BUTTON:
			Input::get_singleton()->joy_button(p_event.device, (JoyButton)p_event.index, p_event.pressed);
			break;
		case JOY_EVENT_AXIS:
			Input::JoyAxisValue value;
			value.min = -1;
			value.value = p_event.value;
			Input::get_singleton()->joy_axis(p_event.device, (JoyAxis)p_event.index, value);
			break;
		case JOY_EVENT_HAT:
			Input::get_singleton()->joy_hat(p_event.device, p_event.hat);
			break;
		default:
			return;
	}
}

void AndroidInputHandler::_set_key_modifier_state(Ref<InputEventWithModifiers> ev) {
	ev->set_shift_pressed(shift_mem);
	ev->set_alt_pressed(alt_mem);
	ev->set_meta_pressed(meta_mem);
	ev->set_ctrl_pressed(control_mem);
}

void AndroidInputHandler::process_key_event(int p_keycode, int p_scancode, int p_unicode_char, bool p_pressed) {
	static char32_t prev_wc = 0;
	char32_t unicode = p_unicode_char;
	if ((p_unicode_char & 0xfffffc00) == 0xd800) {
		if (prev_wc != 0) {
			ERR_PRINT("invalid utf16 surrogate input");
		}
		prev_wc = unicode;
		return; // Skip surrogate.
	} else if ((unicode & 0xfffffc00) == 0xdc00) {
		if (prev_wc == 0) {
			ERR_PRINT("invalid utf16 surrogate input");
			return; // Skip invalid surrogate.
		}
		unicode = (prev_wc << 10UL) + unicode - ((0xd800 << 10UL) + 0xdc00 - 0x10000);
		prev_wc = 0;
	} else {
		prev_wc = 0;
	}

	Ref<InputEventKey> ev;
	ev.instantiate();
	int val = unicode;
	Key keycode = android_get_keysym(p_keycode);
	Key phy_keycode = android_get_keysym(p_scancode);

	if (keycode == Key::SHIFT) {
		shift_mem = p_pressed;
	}
	if (keycode == Key::ALT) {
		alt_mem = p_pressed;
	}
	if (keycode == Key::CTRL) {
		control_mem = p_pressed;
	}
	if (keycode == Key::META) {
		meta_mem = p_pressed;
	}

	ev->set_keycode(keycode);
	ev->set_physical_keycode(phy_keycode);
	ev->set_unicode(val);
	ev->set_pressed(p_pressed);

	_set_key_modifier_state(ev);

	if (val == '\n') {
		ev->set_keycode(Key::ENTER);
	} else if (val == 61448) {
		ev->set_keycode(Key::BACKSPACE);
		ev->set_unicode((char32_t)Key::BACKSPACE);
	} else if (val == 61453) {
		ev->set_keycode(Key::ENTER);
		ev->set_unicode((char32_t)Key::ENTER);
	} else if (p_keycode == 4) {
		if (DisplayServerAndroid *dsa = Object::cast_to<DisplayServerAndroid>(DisplayServer::get_singleton())) {
			dsa->send_window_event(DisplayServer::WINDOW_EVENT_GO_BACK_REQUEST, true);
		}
	}

	Input::get_singleton()->parse_input_event(ev);
}

void AndroidInputHandler::process_touch(int p_event, int p_pointer, const Vector<AndroidInputHandler::TouchPos> &p_points) {
	switch (p_event) {
		case AMOTION_EVENT_ACTION_DOWN: { //gesture begin
			if (touch.size()) {
				//end all if exist
				for (int i = 0; i < touch.size(); i++) {
					Ref<InputEventScreenTouch> ev;
					ev.instantiate();
					ev->set_index(touch[i].id);
					ev->set_pressed(false);
					ev->set_position(touch[i].pos);
					Input::get_singleton()->parse_input_event(ev);
				}
			}

			touch.resize(p_points.size());
			for (int i = 0; i < p_points.size(); i++) {
				touch.write[i].id = p_points[i].id;
				touch.write[i].pos = p_points[i].pos;
			}

			//send touch
			for (int i = 0; i < touch.size(); i++) {
				Ref<InputEventScreenTouch> ev;
				ev.instantiate();
				ev->set_index(touch[i].id);
				ev->set_pressed(true);
				ev->set_position(touch[i].pos);
				Input::get_singleton()->parse_input_event(ev);
			}

		} break;
		case AMOTION_EVENT_ACTION_MOVE: { //motion
			ERR_FAIL_COND(touch.size() != p_points.size());

			for (int i = 0; i < touch.size(); i++) {
				int idx = -1;
				for (int j = 0; j < p_points.size(); j++) {
					if (touch[i].id == p_points[j].id) {
						idx = j;
						break;
					}
				}

				ERR_CONTINUE(idx == -1);

				if (touch[i].pos == p_points[idx].pos)
					continue; //no move unncesearily

				Ref<InputEventScreenDrag> ev;
				ev.instantiate();
				ev->set_index(touch[i].id);
				ev->set_position(p_points[idx].pos);
				ev->set_relative(p_points[idx].pos - touch[i].pos);
				Input::get_singleton()->parse_input_event(ev);
				touch.write[i].pos = p_points[idx].pos;
			}

		} break;
		case AMOTION_EVENT_ACTION_CANCEL:
		case AMOTION_EVENT_ACTION_UP: { //release
			if (touch.size()) {
				//end all if exist
				for (int i = 0; i < touch.size(); i++) {
					Ref<InputEventScreenTouch> ev;
					ev.instantiate();
					ev->set_index(touch[i].id);
					ev->set_pressed(false);
					ev->set_position(touch[i].pos);
					Input::get_singleton()->parse_input_event(ev);
				}
				touch.clear();
			}
		} break;
		case AMOTION_EVENT_ACTION_POINTER_DOWN: { // add touch
			for (int i = 0; i < p_points.size(); i++) {
				if (p_points[i].id == p_pointer) {
					TouchPos tp = p_points[i];
					touch.push_back(tp);

					Ref<InputEventScreenTouch> ev;
					ev.instantiate();

					ev->set_index(tp.id);
					ev->set_pressed(true);
					ev->set_position(tp.pos);
					Input::get_singleton()->parse_input_event(ev);

					break;
				}
			}
		} break;
		case AMOTION_EVENT_ACTION_POINTER_UP: { // remove touch
			for (int i = 0; i < touch.size(); i++) {
				if (touch[i].id == p_pointer) {
					Ref<InputEventScreenTouch> ev;
					ev.instantiate();
					ev->set_index(touch[i].id);
					ev->set_pressed(false);
					ev->set_position(touch[i].pos);
					Input::get_singleton()->parse_input_event(ev);
					touch.remove_at(i);

					break;
				}
			}
		} break;
	}
}

void AndroidInputHandler::process_hover(int p_type, Point2 p_pos) {
	// https://developer.android.com/reference/android/view/MotionEvent.html#ACTION_HOVER_ENTER
	switch (p_type) {
		case AMOTION_EVENT_ACTION_HOVER_MOVE: // hover move
		case AMOTION_EVENT_ACTION_HOVER_ENTER: // hover enter
		case AMOTION_EVENT_ACTION_HOVER_EXIT: { // hover exit
			Ref<InputEventMouseMotion> ev;
			ev.instantiate();
			_set_key_modifier_state(ev);
			ev->set_position(p_pos);
			ev->set_global_position(p_pos);
			ev->set_relative(p_pos - hover_prev_pos);
			Input::get_singleton()->parse_input_event(ev);
			hover_prev_pos = p_pos;
		} break;
	}
}

void AndroidInputHandler::process_mouse_event(int input_device, int event_action, int event_android_buttons_mask, Point2 event_pos, float event_vertical_factor, float event_horizontal_factor) {
	MouseButton event_buttons_mask = _android_button_mask_to_godot_button_mask(event_android_buttons_mask);
	switch (event_action) {
		case AMOTION_EVENT_ACTION_BUTTON_PRESS:
		case AMOTION_EVENT_ACTION_BUTTON_RELEASE: {
			Ref<InputEventMouseButton> ev;
			ev.instantiate();
			_set_key_modifier_state(ev);
			if ((input_device & AINPUT_SOURCE_MOUSE) == AINPUT_SOURCE_MOUSE) {
				ev->set_position(event_pos);
				ev->set_global_position(event_pos);
			} else {
				ev->set_position(hover_prev_pos);
				ev->set_global_position(hover_prev_pos);
			}
			ev->set_pressed(event_action == AMOTION_EVENT_ACTION_BUTTON_PRESS);
			MouseButton changed_button_mask = MouseButton(buttons_state ^ event_buttons_mask);

			buttons_state = event_buttons_mask;

			ev->set_button_index(_button_index_from_mask(changed_button_mask));
			ev->set_button_mask(event_buttons_mask);
			Input::get_singleton()->parse_input_event(ev);
		} break;

		case AMOTION_EVENT_ACTION_MOVE: {
			Ref<InputEventMouseMotion> ev;
			ev.instantiate();
			_set_key_modifier_state(ev);
			if ((input_device & AINPUT_SOURCE_MOUSE) == AINPUT_SOURCE_MOUSE) {
				ev->set_position(event_pos);
				ev->set_global_position(event_pos);
				ev->set_relative(event_pos - hover_prev_pos);
				hover_prev_pos = event_pos;
			} else {
				ev->set_position(hover_prev_pos);
				ev->set_global_position(hover_prev_pos);
				ev->set_relative(event_pos);
			}
			ev->set_button_mask(event_buttons_mask);
			Input::get_singleton()->parse_input_event(ev);
		} break;
		case AMOTION_EVENT_ACTION_SCROLL: {
			Ref<InputEventMouseButton> ev;
			ev.instantiate();
			if ((input_device & AINPUT_SOURCE_MOUSE) == AINPUT_SOURCE_MOUSE) {
				ev->set_position(event_pos);
				ev->set_global_position(event_pos);
			} else {
				ev->set_position(hover_prev_pos);
				ev->set_global_position(hover_prev_pos);
			}
			ev->set_pressed(true);
			buttons_state = event_buttons_mask;
			if (event_vertical_factor > 0) {
				_wheel_button_click(event_buttons_mask, ev, MouseButton::WHEEL_UP, event_vertical_factor);
			} else if (event_vertical_factor < 0) {
				_wheel_button_click(event_buttons_mask, ev, MouseButton::WHEEL_DOWN, -event_vertical_factor);
			}

			if (event_horizontal_factor > 0) {
				_wheel_button_click(event_buttons_mask, ev, MouseButton::WHEEL_RIGHT, event_horizontal_factor);
			} else if (event_horizontal_factor < 0) {
				_wheel_button_click(event_buttons_mask, ev, MouseButton::WHEEL_LEFT, -event_horizontal_factor);
			}
		} break;
	}
}

void AndroidInputHandler::_wheel_button_click(MouseButton event_buttons_mask, const Ref<InputEventMouseButton> &ev, MouseButton wheel_button, float factor) {
	Ref<InputEventMouseButton> evd = ev->duplicate();
	_set_key_modifier_state(evd);
	evd->set_button_index(wheel_button);
	evd->set_button_mask(MouseButton(event_buttons_mask ^ mouse_button_to_mask(wheel_button)));
	evd->set_factor(factor);
	Input::get_singleton()->parse_input_event(evd);
	Ref<InputEventMouseButton> evdd = evd->duplicate();
	evdd->set_pressed(false);
	evdd->set_button_mask(event_buttons_mask);
	Input::get_singleton()->parse_input_event(evdd);
}

void AndroidInputHandler::process_double_tap(int event_android_button_mask, Point2 p_pos) {
	MouseButton event_button_mask = _android_button_mask_to_godot_button_mask(event_android_button_mask);
	Ref<InputEventMouseButton> ev;
	ev.instantiate();
	_set_key_modifier_state(ev);
	ev->set_position(p_pos);
	ev->set_global_position(p_pos);
	ev->set_pressed(event_button_mask != MouseButton::NONE);
	ev->set_button_index(_button_index_from_mask(event_button_mask));
	ev->set_button_mask(event_button_mask);
	ev->set_double_click(true);
	Input::get_singleton()->parse_input_event(ev);
}

MouseButton AndroidInputHandler::_button_index_from_mask(MouseButton button_mask) {
	switch (button_mask) {
		case MouseButton::MASK_LEFT:
			return MouseButton::LEFT;
		case MouseButton::MASK_RIGHT:
			return MouseButton::RIGHT;
		case MouseButton::MASK_MIDDLE:
			return MouseButton::MIDDLE;
		case MouseButton::MASK_XBUTTON1:
			return MouseButton::MB_XBUTTON1;
		case MouseButton::MASK_XBUTTON2:
			return MouseButton::MB_XBUTTON2;
		default:
			return MouseButton::NONE;
	}
}

MouseButton AndroidInputHandler::_android_button_mask_to_godot_button_mask(int android_button_mask) {
	MouseButton godot_button_mask = MouseButton::NONE;
	if (android_button_mask & AMOTION_EVENT_BUTTON_PRIMARY) {
		godot_button_mask |= MouseButton::MASK_LEFT;
	}
	if (android_button_mask & AMOTION_EVENT_BUTTON_SECONDARY) {
		godot_button_mask |= MouseButton::MASK_RIGHT;
	}
	if (android_button_mask & AMOTION_EVENT_BUTTON_TERTIARY) {
		godot_button_mask |= MouseButton::MASK_MIDDLE;
	}
	if (android_button_mask & AMOTION_EVENT_BUTTON_BACK) {
		godot_button_mask |= MouseButton::MASK_XBUTTON1;
	}
	if (android_button_mask & AMOTION_EVENT_BUTTON_FORWARD) {
		godot_button_mask |= MouseButton::MASK_XBUTTON2;
	}

	return godot_button_mask;
}

void AndroidInputHandler::process_scroll(Point2 p_pos) {
	Ref<InputEventPanGesture> ev;
	ev.instantiate();
	_set_key_modifier_state(ev);
	ev->set_position(p_pos);
	ev->set_delta(p_pos - scroll_prev_pos);
	Input::get_singleton()->parse_input_event(ev);
	scroll_prev_pos = p_pos;
}
