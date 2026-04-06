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
#include "display_server_android.h"

#include "core/input/input.h"
#include "core/os/os.h"

void AndroidInputHandler::process_joy_event(AndroidInputHandler::JoypadEvent p_event) {
	switch (p_event.type) {
		case JOY_EVENT_BUTTON:
			Input::get_singleton()->joy_button(p_event.device, (JoyButton)p_event.index, p_event.pressed);
			break;
		case JOY_EVENT_AXIS:
			Input::get_singleton()->joy_axis(p_event.device, (JoyAxis)p_event.index, p_event.value);
			break;
		case JOY_EVENT_HAT:
			Input::get_singleton()->joy_hat(p_event.device, p_event.hat);
			break;
		default:
			return;
	}
}

void AndroidInputHandler::_set_key_modifier_state(Ref<InputEventWithModifiers> ev, Key p_keycode) {
	if (p_keycode != Key::SHIFT) {
		ev->set_shift_pressed(shift_mem);
	}
	if (p_keycode != Key::ALT) {
		ev->set_alt_pressed(alt_mem);
	}
	if (p_keycode != Key::META) {
		ev->set_meta_pressed(meta_mem);
	}
	if (p_keycode != Key::CTRL) {
		ev->set_ctrl_pressed(control_mem);
	}
}

void AndroidInputHandler::process_key_event(int p_physical_keycode, int p_unicode, int p_key_label, bool p_pressed, bool p_echo) {
	static char32_t prev_wc = 0;
	char32_t unicode = p_unicode;
	if ((p_unicode & 0xfffffc00) == 0xd800) {
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

	Key physical_keycode = godot_code_from_android_code(p_physical_keycode);
	Key keycode;
	if (unicode == '\b') { // 0x08
		keycode = Key::BACKSPACE;
	} else if (unicode == '\t') { // 0x09
		keycode = Key::TAB;
	} else if (unicode == '\n') { // 0x0A
		keycode = Key::ENTER;
	} else if (unicode == 0x1B) {
		keycode = Key::ESCAPE;
	} else if (unicode == 0x1F) {
		keycode = Key::KEY_DELETE;
	} else {
		keycode = fix_keycode(unicode, physical_keycode);
	}

	switch (physical_keycode) {
		case Key::SHIFT: {
			shift_mem = p_pressed;
		} break;
		case Key::ALT: {
			alt_mem = p_pressed;
		} break;
		case Key::CTRL: {
			control_mem = p_pressed;
		} break;
		case Key::META: {
			meta_mem = p_pressed;
		} break;
		default:
			break;
	}

	ev->set_keycode(keycode);
	ev->set_physical_keycode(physical_keycode);
	ev->set_key_label(fix_key_label(p_key_label, keycode));
	ev->set_unicode(fix_unicode(unicode));
	ev->set_location(godot_location_from_android_code(p_physical_keycode));
	ev->set_pressed(p_pressed);
	ev->set_echo(p_echo);

	_set_key_modifier_state(ev, keycode);

	if (p_physical_keycode == AKEYCODE_BACK && p_pressed) {
		if (DisplayServerAndroid *dsa = Object::cast_to<DisplayServerAndroid>(DisplayServer::get_singleton())) {
			dsa->send_window_event(DisplayServerEnums::WINDOW_EVENT_GO_BACK_REQUEST, true);
		}
	}

	Input::get_singleton()->parse_input_event(ev);
}

void AndroidInputHandler::_cancel_all_touch() {
	_parse_all_touch(false, true);
	touch.clear();
}

void AndroidInputHandler::_parse_all_touch(bool p_pressed, bool p_canceled, bool p_double_tap) {
	if (touch.size()) {
		//end all if exist
		for (int i = 0; i < touch.size(); i++) {
			Ref<InputEventScreenTouch> ev;
			ev.instantiate();
			ev->set_index(touch[i].id);
			ev->set_pressed(p_pressed);
			ev->set_canceled(p_canceled);
			ev->set_position(touch[i].pos);
			ev->set_double_tap(p_double_tap);
			Input::get_singleton()->parse_input_event(ev);
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
				touch.write[i].pressure = p_points[i].pressure;
				touch.write[i].tilt = p_points[i].tilt;
			}

			//send touch
			_parse_all_touch(true, false, p_double_tap);

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
					continue; // Don't move unnecessarily.
				}

				Ref<InputEventScreenDrag> ev;
				ev.instantiate();
				ev->set_index(touch[i].id);
				ev->set_position(p_points[idx].pos);
				ev->set_relative(p_points[idx].pos - touch[i].pos);
				ev->set_relative_screen_position(ev->get_relative());
				ev->set_pressure(p_points[idx].pressure);
				ev->set_tilt(p_points[idx].tilt);
				Input::get_singleton()->parse_input_event(ev);
				touch.write[i].pos = p_points[idx].pos;
			}

		} break;
		case AMOTION_EVENT_ACTION_CANCEL: {
			_cancel_all_touch();
		} break;
		case AMOTION_EVENT_ACTION_UP: { //release
			_release_all_touch();
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

void AndroidInputHandler::_cancel_mouse_event_info(bool p_source_mouse_relative) {
	buttons_state = BitField<MouseButtonMask>();
	_parse_mouse_event_info(BitField<MouseButtonMask>(), false, true, false, p_source_mouse_relative);
	mouse_event_info.valid = false;
}

void AndroidInputHandler::_parse_mouse_event_info(BitField<MouseButtonMask> event_buttons_mask, bool p_pressed, bool p_canceled, bool p_double_click, bool p_source_mouse_relative) {
	if (!mouse_event_info.valid) {
		return;
	}

	Ref<InputEventMouseButton> ev;
	ev.instantiate();
	_set_key_modifier_state(ev, Key::NONE);
	if (p_source_mouse_relative) {
		ev->set_position(hover_prev_pos);
		ev->set_global_position(hover_prev_pos);
	} else {
		ev->set_position(mouse_event_info.pos);
		ev->set_global_position(mouse_event_info.pos);
		hover_prev_pos = mouse_event_info.pos;
	}
	ev->set_pressed(p_pressed);
	ev->set_canceled(p_canceled);
	BitField<MouseButtonMask> changed_button_mask = buttons_state.get_different(event_buttons_mask);

	buttons_state = event_buttons_mask;

	ev->set_button_index(_button_index_from_mask(changed_button_mask));
	ev->set_button_mask(event_buttons_mask);
	ev->set_double_click(p_double_click);
	Input::get_singleton()->parse_input_event(ev);
}

void AndroidInputHandler::_release_mouse_event_info(bool p_source_mouse_relative) {
	_parse_mouse_event_info(BitField<MouseButtonMask>(), false, false, false, p_source_mouse_relative);
	mouse_event_info.valid = false;
}

void AndroidInputHandler::process_mouse_event(int p_event_action, int p_event_android_buttons_mask, Point2 p_event_pos, Vector2 p_delta, bool p_double_click, bool p_source_mouse_relative, float p_pressure, Vector2 p_tilt) {
	BitField<MouseButtonMask> event_buttons_mask = _android_button_mask_to_godot_button_mask(p_event_android_buttons_mask);
	switch (p_event_action) {
		case AMOTION_EVENT_ACTION_HOVER_MOVE: // hover move
		case AMOTION_EVENT_ACTION_HOVER_ENTER: // hover enter
		case AMOTION_EVENT_ACTION_HOVER_EXIT: { // hover exit
			// https://developer.android.com/reference/android/view/MotionEvent.html#ACTION_HOVER_ENTER
			Ref<InputEventMouseMotion> ev;
			ev.instantiate();
			_set_key_modifier_state(ev, Key::NONE);
			ev->set_position(p_event_pos);
			ev->set_global_position(p_event_pos);
			ev->set_relative(p_event_pos - hover_prev_pos);
			ev->set_relative_screen_position(ev->get_relative());
			Input::get_singleton()->parse_input_event(ev);
			hover_prev_pos = p_event_pos;
		} break;

		case AMOTION_EVENT_ACTION_DOWN:
		case AMOTION_EVENT_ACTION_BUTTON_PRESS: {
			// Release any remaining touches or mouse event
			_release_mouse_event_info();
			_release_all_touch();

			mouse_event_info.valid = true;
			mouse_event_info.pos = p_event_pos;
			_parse_mouse_event_info(event_buttons_mask, true, false, p_double_click, p_source_mouse_relative);
		} break;

		case AMOTION_EVENT_ACTION_CANCEL: {
			_cancel_mouse_event_info(p_source_mouse_relative);
		} break;

		case AMOTION_EVENT_ACTION_UP:
		case AMOTION_EVENT_ACTION_BUTTON_RELEASE: {
			_release_mouse_event_info(p_source_mouse_relative);
		} break;

		case AMOTION_EVENT_ACTION_MOVE: {
			if (!p_source_mouse_relative && !mouse_event_info.valid) {
				return;
			}

			Ref<InputEventMouseMotion> ev;
			ev.instantiate();
			_set_key_modifier_state(ev, Key::NONE);
			if (p_source_mouse_relative) {
				ev->set_position(hover_prev_pos);
				ev->set_global_position(hover_prev_pos);
				ev->set_relative(p_event_pos);
				ev->set_relative_screen_position(p_event_pos);
			} else {
				ev->set_position(p_event_pos);
				ev->set_global_position(p_event_pos);
				ev->set_relative(p_event_pos - hover_prev_pos);
				ev->set_relative_screen_position(ev->get_relative());
				mouse_event_info.pos = p_event_pos;
				hover_prev_pos = p_event_pos;
			}
			ev->set_button_mask(event_buttons_mask);
			ev->set_pressure(p_pressure);
			ev->set_tilt(p_tilt);
			Input::get_singleton()->parse_input_event(ev);
		} break;

		case AMOTION_EVENT_ACTION_SCROLL: {
			Ref<InputEventMouseButton> ev;
			ev.instantiate();
			_set_key_modifier_state(ev, Key::NONE);
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
				_wheel_button_click(event_buttons_mask, ev, MouseButton::WHEEL_UP, p_delta.y);
			} else if (p_delta.y < 0) {
				_wheel_button_click(event_buttons_mask, ev, MouseButton::WHEEL_DOWN, -p_delta.y);
			}

			if (p_delta.x > 0) {
				_wheel_button_click(event_buttons_mask, ev, MouseButton::WHEEL_RIGHT, p_delta.x);
			} else if (p_delta.x < 0) {
				_wheel_button_click(event_buttons_mask, ev, MouseButton::WHEEL_LEFT, -p_delta.x);
			}
		} break;
	}
}

void AndroidInputHandler::_wheel_button_click(BitField<MouseButtonMask> event_buttons_mask, const Ref<InputEventMouseButton> &ev, MouseButton wheel_button, float factor) {
	Ref<InputEventMouseButton> evd = ev->duplicate();
	_set_key_modifier_state(evd, Key::NONE);
	evd->set_button_index(wheel_button);
	evd->set_button_mask(event_buttons_mask.get_different(mouse_button_to_mask(wheel_button)));
	evd->set_factor(factor);
	Input::get_singleton()->parse_input_event(evd);
	Ref<InputEventMouseButton> evdd = evd->duplicate();
	evdd->set_pressed(false);
	evdd->set_button_mask(event_buttons_mask);
	Input::get_singleton()->parse_input_event(evdd);
}

void AndroidInputHandler::process_magnify(Point2 p_pos, float p_factor) {
	Ref<InputEventMagnifyGesture> magnify_event;
	magnify_event.instantiate();
	_set_key_modifier_state(magnify_event, Key::NONE);
	magnify_event->set_position(p_pos);
	magnify_event->set_factor(p_factor);
	Input::get_singleton()->parse_input_event(magnify_event);
}

void AndroidInputHandler::process_pan(Point2 p_pos, Vector2 p_delta) {
	Ref<InputEventPanGesture> pan_event;
	pan_event.instantiate();
	_set_key_modifier_state(pan_event, Key::NONE);
	pan_event->set_position(p_pos);
	pan_event->set_delta(p_delta);
	Input::get_singleton()->parse_input_event(pan_event);
}

MouseButton AndroidInputHandler::_button_index_from_mask(BitField<MouseButtonMask> button_mask) {
	switch (button_mask) {
		case MouseButtonMask::LEFT:
			return MouseButton::LEFT;
		case MouseButtonMask::RIGHT:
			return MouseButton::RIGHT;
		case MouseButtonMask::MIDDLE:
			return MouseButton::MIDDLE;
		case MouseButtonMask::MB_XBUTTON1:
			return MouseButton::MB_XBUTTON1;
		case MouseButtonMask::MB_XBUTTON2:
			return MouseButton::MB_XBUTTON2;
		default:
			return MouseButton::NONE;
	}
}

// Virtual Mouse
void AndroidInputHandler::_vm_emit_mouse_motion(const Point2 &p_finger_pos) {
	Vector2 finger_delta = p_finger_pos - vm_primary_last_pos;
	vm_primary_last_pos = p_finger_pos;

	Vector2 cursor_delta = finger_delta * vm_sensitivity;
	virtual_cursor_pos += cursor_delta;

	Size2 screen = DisplayServer::get_singleton()->screen_get_size();
	virtual_cursor_pos.x = CLAMP(virtual_cursor_pos.x, 0.0f, (float)screen.x);
	virtual_cursor_pos.y = CLAMP(virtual_cursor_pos.y, 0.0f, (float)screen.y);

	Ref<InputEventMouseMotion> ev;
	ev.instantiate();
	_set_key_modifier_state(ev, Key::NONE);
	ev->set_position(virtual_cursor_pos);
	ev->set_global_position(virtual_cursor_pos);
	ev->set_relative(cursor_delta);
	ev->set_relative_screen_position(cursor_delta);

	ev->set_button_mask(buttons_state);

	Input::get_singleton()->parse_input_event(ev);
}

void AndroidInputHandler::_vm_emit_mouse_click(MouseButton p_button, const Point2 &p_pos, bool p_double_click) {
	_vm_emit_mouse_button_press(p_button, p_pos, p_double_click);
	_vm_emit_mouse_button_release(p_button, p_pos);
}

void AndroidInputHandler::_vm_emit_mouse_button_press(MouseButton p_button, const Point2 &p_pos, bool p_double_click) {
	MouseButtonMask mask = mouse_button_to_mask(p_button);
	buttons_state.set_flag(mask);

	Ref<InputEventMouseButton> ev;
	ev.instantiate();
	_set_key_modifier_state(ev, Key::NONE);
	ev->set_position(p_pos);
	ev->set_global_position(p_pos);
	ev->set_button_index(p_button);
	ev->set_button_mask(buttons_state);
	ev->set_pressed(true);
	ev->set_double_click(p_double_click);
	Input::get_singleton()->parse_input_event(ev);
}

void AndroidInputHandler::_vm_emit_mouse_button_release(MouseButton p_button, const Point2 &p_pos) {
	MouseButtonMask mask = mouse_button_to_mask(p_button);
	buttons_state.clear_flag(mask);

	Ref<InputEventMouseButton> ev;
	ev.instantiate();
	_set_key_modifier_state(ev, Key::NONE);
	ev->set_position(p_pos);
	ev->set_global_position(p_pos);
	ev->set_button_index(p_button);
	ev->set_button_mask(buttons_state);
	ev->set_pressed(false);
	Input::get_singleton()->parse_input_event(ev);
}

void AndroidInputHandler::_vm_emit_scroll(float p_delta_x, float p_delta_y) {
	if (Math::abs(p_delta_y) >= 1.0f) {
		MouseButton btn = p_delta_y > 0 ? MouseButton::WHEEL_UP : MouseButton::WHEEL_DOWN;
		float steps = Math::abs(p_delta_y);

		Ref<InputEventMouseButton> ev;
		ev.instantiate();
		_set_key_modifier_state(ev, Key::NONE);
		ev->set_position(virtual_cursor_pos);
		ev->set_global_position(virtual_cursor_pos);
		ev->set_button_index(btn);
		ev->set_button_mask(buttons_state);
		ev->set_factor(steps);
		ev->set_pressed(true);
		Input::get_singleton()->parse_input_event(ev);

		Ref<InputEventMouseButton> ev_up = ev->duplicate();
		ev_up->set_pressed(false);
		Input::get_singleton()->parse_input_event(ev_up);
	}
	if (Math::abs(p_delta_x) >= 1.0f) {
		MouseButton btn = p_delta_x > 0 ? MouseButton::WHEEL_RIGHT : MouseButton::WHEEL_LEFT;
		float steps = Math::abs(p_delta_x);

		Ref<InputEventMouseButton> ev;
		ev.instantiate();
		_set_key_modifier_state(ev, Key::NONE);
		ev->set_position(virtual_cursor_pos);
		ev->set_global_position(virtual_cursor_pos);
		ev->set_button_index(btn);
		ev->set_button_mask(buttons_state);
		ev->set_factor(steps);
		ev->set_pressed(true);
		Input::get_singleton()->parse_input_event(ev);

		Ref<InputEventMouseButton> ev_up = ev->duplicate();
		ev_up->set_pressed(false);
		Input::get_singleton()->parse_input_event(ev_up);
	}
}

void AndroidInputHandler::process_virtual_mouse_touch(int p_event, int p_pointer, const Vector<TouchPos> &p_points) {
	if (!virtual_mouse_enabled) {
		process_touch_event(p_event, p_pointer, p_points, false); //If virtual mouse is not enabled fall back to normal touch
		return;
	}

	switch (p_event) {
		case AMOTION_EVENT_ACTION_DOWN: {
			vm_primary_id = -1;
			vm_secondary_id = -1;
			vm_drag_active = false;
			vm_primary_moved = false;
			vm_scroll_active = false;
			vm_scroll_accum_y = 0.0f;
			vm_scroll_accum_x = 0.0f;

			if (p_points.is_empty()) {
				break;
			}

			vm_primary_id = p_points[0].id;
			vm_primary_start_pos = p_points[0].pos;
			vm_primary_start_ms = OS::get_singleton()->get_ticks_msec();

			vm_primary_last_pos = p_points[0].pos;
		} break;

		case AMOTION_EVENT_ACTION_POINTER_DOWN: {
			if (vm_secondary_id != -1) {
				break;
			}
			if (p_pointer == vm_primary_id) {
				break;
			}

			for (int i = 0; i < p_points.size(); i++) {
				if (p_points[i].id == p_pointer) {
					vm_secondary_start_pos = p_points[i].pos;
					vm_scroll_secondary_last = p_points[i].pos;
					break;
				}
			}

			vm_scroll_primary_last = vm_primary_last_pos;
			vm_scroll_active = false;
			vm_scroll_accum_y = 0.0f;
			vm_scroll_accum_x = 0.0f;
			vm_secondary_id = p_pointer;
			vm_secondary_start_ms = OS::get_singleton()->get_ticks_msec();
			vm_secondary_moved = false;
		} break;

		case AMOTION_EVENT_ACTION_MOVE: {
			if (vm_secondary_id != -1 && !vm_drag_active) {
				Point2 new_primary_pos;
				Point2 new_secondary_pos;
				bool found_primary = false, found_secondary = false;
				for (int i = 0; i < p_points.size(); i++) {
					if (p_points[i].id == vm_primary_id) {
						new_primary_pos = p_points[i].pos;
						found_primary = true;
					}
					if (p_points[i].id == vm_secondary_id) {
						new_secondary_pos = p_points[i].pos;
						found_secondary = true;
					}
				}
				if (found_primary && found_secondary) {
					Vector2 primary_delta = new_primary_pos - vm_scroll_primary_last;
					Vector2 secondary_delta = new_secondary_pos - vm_scroll_secondary_last;

					bool same_dir = primary_delta.dot(secondary_delta) > 0;

					if (same_dir) {
						Vector2 avg_delta = (primary_delta + secondary_delta) * 0.5f;

						if (!vm_scroll_active) {
							if (avg_delta.length() > SCROLL_THRESHOLD_PX * 0.5f) {
								vm_scroll_active = true;
								vm_secondary_moved = true;
							}
						}

						if (vm_scroll_active) {
							vm_scroll_accum_y -= avg_delta.y / SCROLL_PIXELS_PER_STEP;
							vm_scroll_accum_x += avg_delta.x / SCROLL_PIXELS_PER_STEP;
							_vm_emit_scroll(
									(float)(int)vm_scroll_accum_x,
									(float)(int)vm_scroll_accum_y);
							vm_scroll_accum_y -= (float)(int)vm_scroll_accum_y;
							vm_scroll_accum_x -= (float)(int)vm_scroll_accum_x;
						}
					}

					vm_scroll_primary_last = new_primary_pos;
					vm_scroll_secondary_last = new_secondary_pos;
				}
			}

			for (int i = 0; i < p_points.size(); i++) {
				if (p_points[i].id == vm_secondary_id && !vm_secondary_moved) {
					if (vm_secondary_start_pos.distance_to(p_points[i].pos) > TAP_MOVE_THRESHOLD_PX) {
						vm_secondary_moved = true;
					}
				}
				if (p_points[i].id != vm_primary_id) {
					continue;
				}

				Point2 new_finger_pos = p_points[i].pos;

				if (!vm_primary_moved) {
					if (vm_primary_start_pos.distance_to(new_finger_pos) > TAP_MOVE_THRESHOLD_PX) {
						vm_primary_moved = true;
					}
				}

				if (vm_secondary_id != -1 && !vm_drag_active && !vm_scroll_active && vm_primary_moved) {
					vm_drag_active = true;
					_vm_emit_mouse_button_press(MouseButton::LEFT, virtual_cursor_pos);
				}

				if (!vm_scroll_active && new_finger_pos != vm_primary_last_pos) {
					_vm_emit_mouse_motion(new_finger_pos);
				} else if (vm_scroll_active) {
					vm_primary_last_pos = new_finger_pos;
				}
				break;
			}
		} break;

		case AMOTION_EVENT_ACTION_POINTER_UP: {
			if (p_pointer != vm_secondary_id) {
				break;
			}

			if (vm_drag_active) {
				vm_drag_active = false;
				_vm_emit_mouse_button_release(MouseButton::LEFT, virtual_cursor_pos);
			} else if (!vm_secondary_moved) {
				uint64_t held_ms = OS::get_singleton()->get_ticks_msec() - vm_secondary_start_ms;
				if (held_ms < TAP_THRESHOLD_MS) {
					_vm_emit_mouse_click(MouseButton::RIGHT, virtual_cursor_pos);
				}
			}

			vm_secondary_id = -1;
			vm_secondary_moved = false;
		} break;

		case AMOTION_EVENT_ACTION_UP: {
			if (vm_primary_id == -1) {
				break;
			}

			if (vm_drag_active) {
				vm_drag_active = false;
				_vm_emit_mouse_button_release(MouseButton::LEFT, virtual_cursor_pos);
			}

			if (!vm_primary_moved) {
				uint64_t now = OS::get_singleton()->get_ticks_msec();
				uint64_t elapsed = now - vm_primary_start_ms;
				if (elapsed < TAP_THRESHOLD_MS) {
					uint64_t since_last_tap = now - vm_last_tap_ms;
					float dist_from_last = virtual_cursor_pos.distance_to(vm_last_tap_pos);
					bool is_double_click = (since_last_tap < DOUBLE_TAP_THRESHOLD_MS &&
							dist_from_last < DOUBLE_TAP_SLOP_PX);

					_vm_emit_mouse_click(MouseButton::LEFT, virtual_cursor_pos, is_double_click);

					vm_last_tap_ms = now;
					vm_last_tap_pos = virtual_cursor_pos;
				}
			}

			vm_primary_id = -1;
			vm_primary_moved = false;
		} break;

		case AMOTION_EVENT_ACTION_CANCEL: {
			if (vm_drag_active) {
				vm_drag_active = false;
				_vm_emit_mouse_button_release(MouseButton::LEFT, virtual_cursor_pos);
			}

			vm_primary_id = -1;
			vm_secondary_id = -1;
			vm_primary_moved = false;
			vm_scroll_active = false;
			vm_scroll_accum_y = 0.0f;
			vm_scroll_accum_x = 0.0f;
		} break;

		default:
			break;
	}
}

BitField<MouseButtonMask> AndroidInputHandler::_android_button_mask_to_godot_button_mask(int android_button_mask) {
	BitField<MouseButtonMask> godot_button_mask = MouseButtonMask::NONE;
	if (android_button_mask & AMOTION_EVENT_BUTTON_PRIMARY) {
		godot_button_mask.set_flag(MouseButtonMask::LEFT);
	}
	if (android_button_mask & AMOTION_EVENT_BUTTON_SECONDARY) {
		godot_button_mask.set_flag(MouseButtonMask::RIGHT);
	}
	if (android_button_mask & AMOTION_EVENT_BUTTON_TERTIARY) {
		godot_button_mask.set_flag(MouseButtonMask::MIDDLE);
	}
	if (android_button_mask & AMOTION_EVENT_BUTTON_BACK) {
		godot_button_mask.set_flag(MouseButtonMask::MB_XBUTTON1);
	}
	if (android_button_mask & AMOTION_EVENT_BUTTON_FORWARD) {
		godot_button_mask.set_flag(MouseButtonMask::MB_XBUTTON2);
	}

	return godot_button_mask;
}
