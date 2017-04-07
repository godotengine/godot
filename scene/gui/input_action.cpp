/*************************************************************************/
/*  input_action.cpp                                                     */
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
#include "input_action.h"
#include "os/keyboard.h"

void ShortCut::set_shortcut(const InputEvent &p_shortcut) {

	shortcut = p_shortcut;
	emit_changed();
}

InputEvent ShortCut::get_shortcut() const {

	return shortcut;
}

bool ShortCut::is_shortcut(const InputEvent &p_event) const {

	bool same = false;

	switch (p_event.type) {

		case InputEvent::KEY: {

			same = (shortcut.key.scancode == p_event.key.scancode && shortcut.key.mod == p_event.key.mod);

		} break;
		case InputEvent::JOYPAD_BUTTON: {

			same = (shortcut.joy_button.button_index == p_event.joy_button.button_index);

		} break;
		case InputEvent::MOUSE_BUTTON: {

			same = (shortcut.mouse_button.button_index == p_event.mouse_button.button_index);

		} break;
		case InputEvent::JOYPAD_MOTION: {

			same = (shortcut.joy_motion.axis == p_event.joy_motion.axis && (shortcut.joy_motion.axis_value < 0) == (p_event.joy_motion.axis_value < 0));

		} break;
		default: {};
	}

	return same;
}

String ShortCut::get_as_text() const {

	switch (shortcut.type) {

		case InputEvent::NONE: {

			return "None";
		} break;
		case InputEvent::KEY: {

			String str;
			if (shortcut.key.mod.shift)
				str += RTR("Shift+");
			if (shortcut.key.mod.alt)
				str += RTR("Alt+");
			if (shortcut.key.mod.control)
				str += RTR("Ctrl+");
			if (shortcut.key.mod.meta)
				str += RTR("Meta+");

			str += keycode_get_string(shortcut.key.scancode).capitalize();

			return str;
		} break;
		case InputEvent::JOYPAD_BUTTON: {

			String str = RTR("Device") + " " + itos(shortcut.device) + ", " + RTR("Button") + " " + itos(shortcut.joy_button.button_index);
			str += ".";

			return str;
		} break;
		case InputEvent::MOUSE_BUTTON: {

			String str = RTR("Device") + " " + itos(shortcut.device) + ", ";
			switch (shortcut.mouse_button.button_index) {
				case BUTTON_LEFT: str += RTR("Left Button."); break;
				case BUTTON_RIGHT: str += RTR("Right Button."); break;
				case BUTTON_MIDDLE: str += RTR("Middle Button."); break;
				case BUTTON_WHEEL_UP: str += RTR("Wheel Up."); break;
				case BUTTON_WHEEL_DOWN: str += RTR("Wheel Down."); break;
				default: str += RTR("Button") + " " + itos(shortcut.mouse_button.button_index) + ".";
			}

			return str;
		} break;
		case InputEvent::JOYPAD_MOTION: {

			int ax = shortcut.joy_motion.axis;
			String str = RTR("Device") + " " + itos(shortcut.device) + ", " + RTR("Axis") + " " + itos(ax) + ".";

			return str;
		} break;
	}

	return "";
}

bool ShortCut::is_valid() const {

	return shortcut.type != InputEvent::NONE;
}

void ShortCut::_bind_methods() {

	ClassDB::bind_method(D_METHOD("set_shortcut", "event"), &ShortCut::set_shortcut);
	ClassDB::bind_method(D_METHOD("get_shortcut"), &ShortCut::get_shortcut);

	ClassDB::bind_method(D_METHOD("is_valid"), &ShortCut::is_valid);

	ClassDB::bind_method(D_METHOD("is_shortcut", "event"), &ShortCut::is_shortcut);
	ClassDB::bind_method(D_METHOD("get_as_text"), &ShortCut::get_as_text);

	ADD_PROPERTY(PropertyInfo(Variant::INPUT_EVENT, "shortcut"), "set_shortcut", "get_shortcut");
}

ShortCut::ShortCut() {
}
