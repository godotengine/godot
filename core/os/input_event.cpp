/*************************************************************************/
/*  input_event.cpp                                                      */
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
#include "input_event.h"
#include "input_map.h"
#include "os/keyboard.h"
/**
 *
 */

bool InputEvent::operator==(const InputEvent &p_event) const {
	if (type != p_event.type) {
		return false;
	}

	switch (type) {
		/** Current clang-format style doesn't play well with the aligned return values of that switch. */
		/* clang-format off */
		case NONE:
			return true;
		case KEY:
			return key.unicode == p_event.key.unicode
				&& key.scancode == p_event.key.scancode
				&& key.echo == p_event.key.echo
				&& key.pressed == p_event.key.pressed
				&& key.mod == p_event.key.mod;
		case MOUSE_MOTION:
			return mouse_motion.x == p_event.mouse_motion.x
				&& mouse_motion.y == p_event.mouse_motion.y
				&& mouse_motion.relative_x == p_event.mouse_motion.relative_x
				&& mouse_motion.relative_y == p_event.mouse_motion.relative_y
				&& mouse_motion.button_mask == p_event.mouse_motion.button_mask
				&& key.mod == p_event.key.mod;
		case MOUSE_BUTTON:
			return mouse_button.pressed == p_event.mouse_button.pressed
				&& mouse_button.x == p_event.mouse_button.x
				&& mouse_button.y == p_event.mouse_button.y
				&& mouse_button.button_index == p_event.mouse_button.button_index
				&& mouse_button.button_mask == p_event.mouse_button.button_mask
				&& key.mod == p_event.key.mod;
		case JOYPAD_MOTION:
			return joy_motion.axis == p_event.joy_motion.axis
				&& joy_motion.axis_value == p_event.joy_motion.axis_value;
		case JOYPAD_BUTTON:
			return joy_button.pressed == p_event.joy_button.pressed
				&& joy_button.button_index == p_event.joy_button.button_index
				&& joy_button.pressure == p_event.joy_button.pressure;
		case SCREEN_TOUCH:
			return screen_touch.pressed == p_event.screen_touch.pressed
				&& screen_touch.index == p_event.screen_touch.index
				&& screen_touch.x == p_event.screen_touch.x
				&& screen_touch.y == p_event.screen_touch.y;
		case SCREEN_DRAG:
			return screen_drag.index == p_event.screen_drag.index
				&& screen_drag.x == p_event.screen_drag.x
				&& screen_drag.y == p_event.screen_drag.y;
		case ACTION:
			return action.action == p_event.action.action
				&& action.pressed == p_event.action.pressed;
		/* clang-format on */
		default:
			ERR_PRINT("No logic to compare InputEvents of this type, this shouldn't happen.");
	}

	return false;
}
InputEvent::operator String() const {

	String str = "Device " + itos(device) + " ID " + itos(ID) + " ";

	switch (type) {

		case NONE: {

			return "Event: None";
		} break;
		case KEY: {

			str += "Event: Key ";
			str = str + "Unicode: " + String::chr(key.unicode) + " Scan: " + itos(key.scancode) + " Echo: " + String(key.echo ? "True" : "False") + " Pressed" + String(key.pressed ? "True" : "False") + " Mod: ";
			if (key.mod.shift)
				str += "S";
			if (key.mod.control)
				str += "C";
			if (key.mod.alt)
				str += "A";
			if (key.mod.meta)
				str += "M";

			return str;
		} break;
		case MOUSE_MOTION: {

			str += "Event: Motion ";
			str = str + " Pos: " + itos(mouse_motion.x) + "," + itos(mouse_motion.y) + " Rel: " + itos(mouse_motion.relative_x) + "," + itos(mouse_motion.relative_y) + " Mask: ";
			for (int i = 0; i < 8; i++) {

				if ((1 << i) & mouse_motion.button_mask)
					str += itos(i + 1);
			}
			str += " Mod: ";
			if (key.mod.shift)
				str += "S";
			if (key.mod.control)
				str += "C";
			if (key.mod.alt)
				str += "A";
			if (key.mod.meta)
				str += "M";

			return str;
		} break;
		case MOUSE_BUTTON: {
			str += "Event: Button ";
			str = str + "Pressed: " + itos(mouse_button.pressed) + " Pos: " + itos(mouse_button.x) + "," + itos(mouse_button.y) + " Button: " + itos(mouse_button.button_index) + " Mask: ";
			for (int i = 0; i < 8; i++) {

				if ((1 << i) & mouse_button.button_mask)
					str += itos(i + 1);
			}
			str += " Mod: ";
			if (key.mod.shift)
				str += "S";
			if (key.mod.control)
				str += "C";
			if (key.mod.alt)
				str += "A";
			if (key.mod.meta)
				str += "M";

			str += String(" DoubleClick: ") + (mouse_button.doubleclick ? "Yes" : "No");

			return str;

		} break;
		case JOYPAD_MOTION: {
			str += "Event: JoypadMotion ";
			str = str + "Axis: " + itos(joy_motion.axis) + " Value: " + rtos(joy_motion.axis_value);
			return str;

		} break;
		case JOYPAD_BUTTON: {
			str += "Event: JoypadButton ";
			str = str + "Pressed: " + itos(joy_button.pressed) + " Index: " + itos(joy_button.button_index) + " pressure " + rtos(joy_button.pressure);
			return str;

		} break;
		case SCREEN_TOUCH: {
			str += "Event: ScreenTouch ";
			str = str + "Pressed: " + itos(screen_touch.pressed) + " Index: " + itos(screen_touch.index) + " pos " + rtos(screen_touch.x) + "," + rtos(screen_touch.y);
			return str;

		} break;
		case SCREEN_DRAG: {
			str += "Event: ScreenDrag ";
			str = str + " Index: " + itos(screen_drag.index) + " pos " + rtos(screen_drag.x) + "," + rtos(screen_drag.y);
			return str;

		} break;
		case ACTION: {
			str += "Event: Action: " + InputMap::get_singleton()->get_action_from_id(action.action) + " Pressed: " + itos(action.pressed);
			return str;

		} break;
	}

	return "";
}

void InputEvent::set_as_action(const String &p_action, bool p_pressed) {

	type = ACTION;
	action.action = InputMap::get_singleton()->get_action_id(p_action);
	action.pressed = p_pressed;
}

bool InputEvent::is_pressed() const {

	switch (type) {

		case KEY: return key.pressed;
		case MOUSE_BUTTON: return mouse_button.pressed;
		case JOYPAD_BUTTON: return joy_button.pressed;
		case SCREEN_TOUCH: return screen_touch.pressed;
		case JOYPAD_MOTION: return ABS(joy_motion.axis_value) > 0.5;
		case ACTION: return action.pressed;
		default: {}
	}

	return false;
}

bool InputEvent::is_echo() const {

	return (type == KEY && key.echo);
}

bool InputEvent::is_action(const String &p_action) const {

	return InputMap::get_singleton()->event_is_action(*this, p_action);
}

bool InputEvent::is_action_pressed(const String &p_action) const {

	return is_action(p_action) && is_pressed() && !is_echo();
}

bool InputEvent::is_action_released(const String &p_action) const {

	return is_action(p_action) && !is_pressed();
}

uint32_t InputEventKey::get_scancode_with_modifiers() const {

	uint32_t sc = scancode;
	if (mod.control)
		sc |= KEY_MASK_CTRL;
	if (mod.alt)
		sc |= KEY_MASK_ALT;
	if (mod.shift)
		sc |= KEY_MASK_SHIFT;
	if (mod.meta)
		sc |= KEY_MASK_META;

	return sc;
}

InputEvent InputEvent::xform_by(const Transform2D &p_xform) const {

	InputEvent ev = *this;

	switch (ev.type) {

		case InputEvent::MOUSE_BUTTON: {

			Vector2 g = p_xform.xform(Vector2(ev.mouse_button.global_x, ev.mouse_button.global_y));
			Vector2 l = p_xform.xform(Vector2(ev.mouse_button.x, ev.mouse_button.y));
			ev.mouse_button.x = l.x;
			ev.mouse_button.y = l.y;
			ev.mouse_button.global_x = g.x;
			ev.mouse_button.global_y = g.y;

		} break;
		case InputEvent::MOUSE_MOTION: {

			Vector2 g = p_xform.xform(Vector2(ev.mouse_motion.global_x, ev.mouse_motion.global_y));
			Vector2 l = p_xform.xform(Vector2(ev.mouse_motion.x, ev.mouse_motion.y));
			Vector2 r = p_xform.basis_xform(Vector2(ev.mouse_motion.relative_x, ev.mouse_motion.relative_y));
			Vector2 s = p_xform.basis_xform(Vector2(ev.mouse_motion.speed_x, ev.mouse_motion.speed_y));
			ev.mouse_motion.x = l.x;
			ev.mouse_motion.y = l.y;
			ev.mouse_motion.global_x = g.x;
			ev.mouse_motion.global_y = g.y;
			ev.mouse_motion.relative_x = r.x;
			ev.mouse_motion.relative_y = r.y;
			ev.mouse_motion.speed_x = s.x;
			ev.mouse_motion.speed_y = s.y;

		} break;
		case InputEvent::SCREEN_TOUCH: {

			Vector2 t = p_xform.xform(Vector2(ev.screen_touch.x, ev.screen_touch.y));
			ev.screen_touch.x = t.x;
			ev.screen_touch.y = t.y;

		} break;
		case InputEvent::SCREEN_DRAG: {

			Vector2 t = p_xform.xform(Vector2(ev.screen_drag.x, ev.screen_drag.y));
			Vector2 r = p_xform.basis_xform(Vector2(ev.screen_drag.relative_x, ev.screen_drag.relative_y));
			Vector2 s = p_xform.basis_xform(Vector2(ev.screen_drag.speed_x, ev.screen_drag.speed_y));
			ev.screen_drag.x = t.x;
			ev.screen_drag.y = t.y;
			ev.screen_drag.relative_x = r.x;
			ev.screen_drag.relative_y = r.y;
			ev.screen_drag.speed_x = s.x;
			ev.screen_drag.speed_y = s.y;
		} break;
	}

	return ev;
}
