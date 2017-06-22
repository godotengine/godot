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

void ShortCut::set_shortcut(const Ref<InputEvent> &p_shortcut) {

	shortcut = p_shortcut;
	emit_changed();
}

Ref<InputEvent> ShortCut::get_shortcut() const {

	return shortcut;
}

bool ShortCut::is_shortcut(const Ref<InputEvent> &p_event) const {

	return shortcut.is_valid() && shortcut->shortcut_match(p_event);
}

String ShortCut::get_as_text() const {

	if (shortcut.is_valid())
		return shortcut->as_text();
	else
		return "None";
#if 0
	switch (shortcut.type) {

		case Ref<InputEvent>::NONE: {

			return "None";
		} break;
		case Ref<InputEvent>::KEY: {

			String str;
			if (shortcut->get_shift())
				str += RTR("Shift+");
			if (shortcut->get_alt())
				str += RTR("Alt+");
			if (shortcut->get_control())
				str += RTR("Ctrl+");
			if (shortcut->get_metakey())
				str += RTR("Meta+");

			str += keycode_get_string(shortcut->get_scancode()).capitalize();

			return str;
		} break;
		case Ref<InputEvent>::JOYPAD_BUTTON: {

			String str = RTR("Device") + " " + itos(shortcut.device) + ", " + RTR("Button") + " " + itos(shortcut.joy_button->get_button_index());
			str += ".";

			return str;
		} break;
		case Ref<InputEvent>::MOUSE_BUTTON: {

			String str = RTR("Device") + " " + itos(shortcut.device) + ", ";
			switch (shortcut->get_button_index()) {
				case BUTTON_LEFT: str += RTR("Left Button."); break;
				case BUTTON_RIGHT: str += RTR("Right Button."); break;
				case BUTTON_MIDDLE: str += RTR("Middle Button."); break;
				case BUTTON_WHEEL_UP: str += RTR("Wheel Up."); break;
				case BUTTON_WHEEL_DOWN: str += RTR("Wheel Down."); break;
				default: str += RTR("Button") + " " + itos(shortcut->get_button_index()) + ".";
			}

			return str;
		} break;
		case Ref<InputEvent>::JOYPAD_MOTION: {

			int ax = shortcut.joy_motion.axis;
			String str = RTR("Device") + " " + itos(shortcut.device) + ", " + RTR("Axis") + " " + itos(ax) + ".";

			return str;
		} break;
	}

	return "";
#endif
}

bool ShortCut::is_valid() const {

	return shortcut.is_valid();
}

void ShortCut::_bind_methods() {

	ClassDB::bind_method(D_METHOD("set_shortcut", "event:InputEvent"), &ShortCut::set_shortcut);
	ClassDB::bind_method(D_METHOD("get_shortcut:InputEvent"), &ShortCut::get_shortcut);

	ClassDB::bind_method(D_METHOD("is_valid"), &ShortCut::is_valid);

	ClassDB::bind_method(D_METHOD("is_shortcut", "event:InputEvent"), &ShortCut::is_shortcut);
	ClassDB::bind_method(D_METHOD("get_as_text"), &ShortCut::get_as_text);

	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "shortcut", PROPERTY_HINT_RESOURCE_TYPE, "InputEvent"), "set_shortcut", "get_shortcut");
}

ShortCut::ShortCut() {
}
