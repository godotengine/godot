/*************************************************************************/
/*  input_event.cpp                                                      */
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

#include "input_event.h"

#include "core/input/input_map.h"
#include "core/input/shortcut.h"
#include "core/os/keyboard.h"

const int InputEvent::DEVICE_ID_TOUCH_MOUSE = -1;
const int InputEvent::DEVICE_ID_INTERNAL = -2;

void InputEvent::set_device(int p_device) {
	device = p_device;
	emit_changed();
}

int InputEvent::get_device() const {
	return device;
}

bool InputEvent::is_action(const StringName &p_action, bool p_exact_match) const {
	return InputMap::get_singleton()->event_is_action(Ref<InputEvent>((InputEvent *)this), p_action, p_exact_match);
}

bool InputEvent::is_action_pressed(const StringName &p_action, bool p_allow_echo, bool p_exact_match) const {
	bool pressed;
	bool valid = InputMap::get_singleton()->event_get_action_status(Ref<InputEvent>((InputEvent *)this), p_action, p_exact_match, &pressed, nullptr, nullptr);
	return valid && pressed && (p_allow_echo || !is_echo());
}

bool InputEvent::is_action_released(const StringName &p_action, bool p_exact_match) const {
	bool pressed;
	bool valid = InputMap::get_singleton()->event_get_action_status(Ref<InputEvent>((InputEvent *)this), p_action, p_exact_match, &pressed, nullptr, nullptr);
	return valid && !pressed;
}

float InputEvent::get_action_strength(const StringName &p_action, bool p_exact_match) const {
	float strength;
	bool valid = InputMap::get_singleton()->event_get_action_status(Ref<InputEvent>((InputEvent *)this), p_action, p_exact_match, nullptr, &strength, nullptr);
	return valid ? strength : 0.0f;
}

float InputEvent::get_action_raw_strength(const StringName &p_action, bool p_exact_match) const {
	float raw_strength;
	bool valid = InputMap::get_singleton()->event_get_action_status(Ref<InputEvent>((InputEvent *)this), p_action, p_exact_match, nullptr, nullptr, &raw_strength);
	return valid ? raw_strength : 0.0f;
}

bool InputEvent::is_pressed() const {
	return false;
}

bool InputEvent::is_echo() const {
	return false;
}

Ref<InputEvent> InputEvent::xformed_by(const Transform2D &p_xform, const Vector2 &p_local_ofs) const {
	return Ref<InputEvent>((InputEvent *)this);
}

bool InputEvent::action_match(const Ref<InputEvent> &p_event, bool *p_pressed, float *p_strength, float *p_raw_strength, float p_deadzone) const {
	return false;
}

bool InputEvent::is_match(const Ref<InputEvent> &p_event, bool p_exact_match) const {
	return false;
}

bool InputEvent::is_action_type() const {
	return false;
}

void InputEvent::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_device", "device"), &InputEvent::set_device);
	ClassDB::bind_method(D_METHOD("get_device"), &InputEvent::get_device);

	ClassDB::bind_method(D_METHOD("is_action", "action", "exact_match"), &InputEvent::is_action, DEFVAL(false));
	ClassDB::bind_method(D_METHOD("is_action_pressed", "action", "allow_echo", "exact_match"), &InputEvent::is_action_pressed, DEFVAL(false), DEFVAL(false));
	ClassDB::bind_method(D_METHOD("is_action_released", "action", "exact_match"), &InputEvent::is_action_released, DEFVAL(false));
	ClassDB::bind_method(D_METHOD("get_action_strength", "action", "exact_match"), &InputEvent::get_action_strength, DEFVAL(false));

	ClassDB::bind_method(D_METHOD("is_pressed"), &InputEvent::is_pressed);
	ClassDB::bind_method(D_METHOD("is_echo"), &InputEvent::is_echo);

	ClassDB::bind_method(D_METHOD("as_text"), &InputEvent::as_text);

	ClassDB::bind_method(D_METHOD("is_match", "event", "exact_match"), &InputEvent::is_match, DEFVAL(true));

	ClassDB::bind_method(D_METHOD("is_action_type"), &InputEvent::is_action_type);

	ClassDB::bind_method(D_METHOD("accumulate", "with_event"), &InputEvent::accumulate);

	ClassDB::bind_method(D_METHOD("xformed_by", "xform", "local_ofs"), &InputEvent::xformed_by, DEFVAL(Vector2()));

	ADD_PROPERTY(PropertyInfo(Variant::INT, "device"), "set_device", "get_device");
}

///////////////////////////////////

void InputEventFromWindow::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_window_id", "id"), &InputEventFromWindow::set_window_id);
	ClassDB::bind_method(D_METHOD("get_window_id"), &InputEventFromWindow::get_window_id);
	ADD_PROPERTY(PropertyInfo(Variant::INT, "window_id"), "set_window_id", "get_window_id");
}

void InputEventFromWindow::set_window_id(int64_t p_id) {
	window_id = p_id;
	emit_changed();
}

int64_t InputEventFromWindow::get_window_id() const {
	return window_id;
}

///////////////////////////////////

void InputEventWithModifiers::set_store_command(bool p_enabled) {
	store_command = p_enabled;
	emit_changed();
}

bool InputEventWithModifiers::is_storing_command() const {
	return store_command;
}

void InputEventWithModifiers::set_shift_pressed(bool p_enabled) {
	shift_pressed = p_enabled;
	emit_changed();
}

bool InputEventWithModifiers::is_shift_pressed() const {
	return shift_pressed;
}

void InputEventWithModifiers::set_alt_pressed(bool p_enabled) {
	alt_pressed = p_enabled;
	emit_changed();
}

bool InputEventWithModifiers::is_alt_pressed() const {
	return alt_pressed;
}

void InputEventWithModifiers::set_ctrl_pressed(bool p_enabled) {
	ctrl_pressed = p_enabled;
	emit_changed();
}

bool InputEventWithModifiers::is_ctrl_pressed() const {
	return ctrl_pressed;
}

void InputEventWithModifiers::set_meta_pressed(bool p_enabled) {
	meta_pressed = p_enabled;
	emit_changed();
}

bool InputEventWithModifiers::is_meta_pressed() const {
	return meta_pressed;
}

void InputEventWithModifiers::set_command_pressed(bool p_enabled) {
	command_pressed = p_enabled;
	emit_changed();
}

bool InputEventWithModifiers::is_command_pressed() const {
	return command_pressed;
}

void InputEventWithModifiers::set_modifiers_from_event(const InputEventWithModifiers *event) {
	set_alt_pressed(event->is_alt_pressed());
	set_shift_pressed(event->is_shift_pressed());
	set_ctrl_pressed(event->is_ctrl_pressed());
	set_meta_pressed(event->is_meta_pressed());
}

uint32_t InputEventWithModifiers::get_modifiers_mask() const {
	uint32_t mask = 0;
	if (is_ctrl_pressed()) {
		mask |= KEY_MASK_CTRL;
	}
	if (is_shift_pressed()) {
		mask |= KEY_MASK_SHIFT;
	}
	if (is_alt_pressed()) {
		mask |= KEY_MASK_ALT;
	}
	if (is_meta_pressed()) {
		mask |= KEY_MASK_META;
	}
	return mask;
}

String InputEventWithModifiers::as_text() const {
	Vector<String> mod_names;

	if (is_ctrl_pressed()) {
		mod_names.push_back(find_keycode_name(KEY_CTRL));
	}
	if (is_shift_pressed()) {
		mod_names.push_back(find_keycode_name(KEY_SHIFT));
	}
	if (is_alt_pressed()) {
		mod_names.push_back(find_keycode_name(KEY_ALT));
	}
	if (is_meta_pressed()) {
		mod_names.push_back(find_keycode_name(KEY_META));
	}

	if (!mod_names.is_empty()) {
		return String("+").join(mod_names);
	} else {
		return "";
	}
}

String InputEventWithModifiers::to_string() {
	return as_text();
}

void InputEventWithModifiers::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_store_command", "enable"), &InputEventWithModifiers::set_store_command);
	ClassDB::bind_method(D_METHOD("is_storing_command"), &InputEventWithModifiers::is_storing_command);

	ClassDB::bind_method(D_METHOD("set_alt_pressed", "pressed"), &InputEventWithModifiers::set_alt_pressed);
	ClassDB::bind_method(D_METHOD("is_alt_pressed"), &InputEventWithModifiers::is_alt_pressed);

	ClassDB::bind_method(D_METHOD("set_shift_pressed", "pressed"), &InputEventWithModifiers::set_shift_pressed);
	ClassDB::bind_method(D_METHOD("is_shift_pressed"), &InputEventWithModifiers::is_shift_pressed);

	ClassDB::bind_method(D_METHOD("set_ctrl_pressed", "pressed"), &InputEventWithModifiers::set_ctrl_pressed);
	ClassDB::bind_method(D_METHOD("is_ctrl_pressed"), &InputEventWithModifiers::is_ctrl_pressed);

	ClassDB::bind_method(D_METHOD("set_meta_pressed", "pressed"), &InputEventWithModifiers::set_meta_pressed);
	ClassDB::bind_method(D_METHOD("is_meta_pressed"), &InputEventWithModifiers::is_meta_pressed);

	ClassDB::bind_method(D_METHOD("set_command_pressed", "pressed"), &InputEventWithModifiers::set_command_pressed);
	ClassDB::bind_method(D_METHOD("is_command_pressed"), &InputEventWithModifiers::is_command_pressed);

	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "store_command"), "set_store_command", "is_storing_command");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "alt_pressed"), "set_alt_pressed", "is_alt_pressed");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "shift_pressed"), "set_shift_pressed", "is_shift_pressed");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "ctrl_pressed"), "set_ctrl_pressed", "is_ctrl_pressed");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "meta_pressed"), "set_meta_pressed", "is_meta_pressed");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "command_pressed"), "set_command_pressed", "is_command_pressed");
}

void InputEventWithModifiers::_validate_property(PropertyInfo &property) const {
	if (store_command) {
		// If we only want to Store "Command".
#ifdef APPLE_STYLE_KEYS
		// Don't store "Meta" on Mac.
		if (property.name == "meta_pressed") {
			property.usage ^= PROPERTY_USAGE_STORAGE;
		}
#else
		// Don't store "Ctrl".
		if (property.name == "ctrl_pressed") {
			property.usage ^= PROPERTY_USAGE_STORAGE;
		}
#endif
	} else {
		// We don't want to store command, only ctrl or meta (on mac).
		if (property.name == "command_pressed") {
			property.usage ^= PROPERTY_USAGE_STORAGE;
		}
	}
}

///////////////////////////////////

void InputEventKey::set_pressed(bool p_pressed) {
	pressed = p_pressed;
	emit_changed();
}

bool InputEventKey::is_pressed() const {
	return pressed;
}

void InputEventKey::set_keycode(Key p_keycode) {
	keycode = p_keycode;
	emit_changed();
}

Key InputEventKey::get_keycode() const {
	return keycode;
}

void InputEventKey::set_physical_keycode(Key p_keycode) {
	physical_keycode = p_keycode;
	emit_changed();
}

Key InputEventKey::get_physical_keycode() const {
	return physical_keycode;
}

void InputEventKey::set_unicode(uint32_t p_unicode) {
	unicode = p_unicode;
	emit_changed();
}

uint32_t InputEventKey::get_unicode() const {
	return unicode;
}

void InputEventKey::set_echo(bool p_enable) {
	echo = p_enable;
	emit_changed();
}

bool InputEventKey::is_echo() const {
	return echo;
}

uint32_t InputEventKey::get_keycode_with_modifiers() const {
	return keycode | get_modifiers_mask();
}

uint32_t InputEventKey::get_physical_keycode_with_modifiers() const {
	return physical_keycode | get_modifiers_mask();
}

String InputEventKey::as_text() const {
	String kc;

	if (keycode == 0) {
		kc = keycode_get_string(physical_keycode) + " (" + RTR("Physical") + ")";
	} else {
		kc = keycode_get_string(keycode);
	}

	if (kc == String()) {
		return kc;
	}

	String mods_text = InputEventWithModifiers::as_text();
	return mods_text == "" ? kc : mods_text + "+" + kc;
}

String InputEventKey::to_string() {
	String p = is_pressed() ? "true" : "false";
	String e = is_echo() ? "true" : "false";

	String kc = "";
	String physical = "false";
	if (keycode == 0) {
		kc = itos(physical_keycode) + " (" + keycode_get_string(physical_keycode) + ")";
		physical = "true";
	} else {
		kc = itos(keycode) + " (" + keycode_get_string(keycode) + ")";
	}

	String mods = InputEventWithModifiers::as_text();
	mods = mods == "" ? TTR("none") : mods;

	return vformat("InputEventKey: keycode=%s, mods=%s, physical=%s, pressed=%s, echo=%s", kc, mods, physical, p, e);
}

Ref<InputEventKey> InputEventKey::create_reference(Key p_keycode) {
	Ref<InputEventKey> ie;
	ie.instantiate();
	ie->set_keycode(p_keycode & KEY_CODE_MASK);
	ie->set_unicode(p_keycode & KEY_CODE_MASK);

	if (p_keycode & KEY_MASK_SHIFT) {
		ie->set_shift_pressed(true);
	}
	if (p_keycode & KEY_MASK_ALT) {
		ie->set_alt_pressed(true);
	}
	if (p_keycode & KEY_MASK_CTRL) {
		ie->set_ctrl_pressed(true);
	}
	if (p_keycode & KEY_MASK_CMD) {
		ie->set_command_pressed(true);
	}
	if (p_keycode & KEY_MASK_META) {
		ie->set_meta_pressed(true);
	}

	return ie;
}

bool InputEventKey::action_match(const Ref<InputEvent> &p_event, bool *p_pressed, float *p_strength, float *p_raw_strength, float p_deadzone) const {
	Ref<InputEventKey> key = p_event;
	if (key.is_null()) {
		return false;
	}

	bool match = false;
	if (get_keycode() == 0) {
		uint32_t code = get_physical_keycode_with_modifiers();
		uint32_t event_code = key->get_physical_keycode_with_modifiers();

		match = get_physical_keycode() == key->get_physical_keycode() && (!key->is_pressed() || (code & event_code) == code);
	} else {
		uint32_t code = get_keycode_with_modifiers();
		uint32_t event_code = key->get_keycode_with_modifiers();

		match = get_keycode() == key->get_keycode() && (!key->is_pressed() || (code & event_code) == code);
	}
	if (match) {
		bool pressed = key->is_pressed();
		if (p_pressed != nullptr) {
			*p_pressed = pressed;
		}
		float strength = pressed ? 1.0f : 0.0f;
		if (p_strength != nullptr) {
			*p_strength = strength;
		}
		if (p_raw_strength != nullptr) {
			*p_raw_strength = strength;
		}
	}
	return match;
}

bool InputEventKey::is_match(const Ref<InputEvent> &p_event, bool p_exact_match) const {
	Ref<InputEventKey> key = p_event;
	if (key.is_null()) {
		return false;
	}

	if (keycode == 0) {
		return physical_keycode == key->physical_keycode &&
				(!p_exact_match || get_modifiers_mask() == key->get_modifiers_mask());
	} else {
		return keycode == key->keycode &&
				(!p_exact_match || get_modifiers_mask() == key->get_modifiers_mask());
	}
}

void InputEventKey::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_pressed", "pressed"), &InputEventKey::set_pressed);

	ClassDB::bind_method(D_METHOD("set_keycode", "keycode"), &InputEventKey::set_keycode);
	ClassDB::bind_method(D_METHOD("get_keycode"), &InputEventKey::get_keycode);

	ClassDB::bind_method(D_METHOD("set_physical_keycode", "physical_keycode"), &InputEventKey::set_physical_keycode);
	ClassDB::bind_method(D_METHOD("get_physical_keycode"), &InputEventKey::get_physical_keycode);

	ClassDB::bind_method(D_METHOD("set_unicode", "unicode"), &InputEventKey::set_unicode);
	ClassDB::bind_method(D_METHOD("get_unicode"), &InputEventKey::get_unicode);

	ClassDB::bind_method(D_METHOD("set_echo", "echo"), &InputEventKey::set_echo);

	ClassDB::bind_method(D_METHOD("get_keycode_with_modifiers"), &InputEventKey::get_keycode_with_modifiers);
	ClassDB::bind_method(D_METHOD("get_physical_keycode_with_modifiers"), &InputEventKey::get_physical_keycode_with_modifiers);

	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "pressed"), "set_pressed", "is_pressed");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "keycode"), "set_keycode", "get_keycode");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "physical_keycode"), "set_physical_keycode", "get_physical_keycode");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "unicode"), "set_unicode", "get_unicode");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "echo"), "set_echo", "is_echo");
}

///////////////////////////////////

void InputEventMouse::set_button_mask(int p_mask) {
	button_mask = p_mask;
	emit_changed();
}

int InputEventMouse::get_button_mask() const {
	return button_mask;
}

void InputEventMouse::set_position(const Vector2 &p_pos) {
	pos = p_pos;
}

Vector2 InputEventMouse::get_position() const {
	return pos;
}

void InputEventMouse::set_global_position(const Vector2 &p_global_pos) {
	global_pos = p_global_pos;
}

Vector2 InputEventMouse::get_global_position() const {
	return global_pos;
}

void InputEventMouse::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_button_mask", "button_mask"), &InputEventMouse::set_button_mask);
	ClassDB::bind_method(D_METHOD("get_button_mask"), &InputEventMouse::get_button_mask);

	ClassDB::bind_method(D_METHOD("set_position", "position"), &InputEventMouse::set_position);
	ClassDB::bind_method(D_METHOD("get_position"), &InputEventMouse::get_position);

	ClassDB::bind_method(D_METHOD("set_global_position", "global_position"), &InputEventMouse::set_global_position);
	ClassDB::bind_method(D_METHOD("get_global_position"), &InputEventMouse::get_global_position);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "button_mask"), "set_button_mask", "get_button_mask");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR2, "position"), "set_position", "get_position");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR2, "global_position"), "set_global_position", "get_global_position");
}

///////////////////////////////////

void InputEventMouseButton::set_factor(float p_factor) {
	factor = p_factor;
}

float InputEventMouseButton::get_factor() const {
	return factor;
}

void InputEventMouseButton::set_button_index(MouseButton p_index) {
	button_index = p_index;
	emit_changed();
}

MouseButton InputEventMouseButton::get_button_index() const {
	return button_index;
}

void InputEventMouseButton::set_pressed(bool p_pressed) {
	pressed = p_pressed;
}

bool InputEventMouseButton::is_pressed() const {
	return pressed;
}

void InputEventMouseButton::set_double_click(bool p_double_click) {
	double_click = p_double_click;
}

bool InputEventMouseButton::is_double_click() const {
	return double_click;
}

Ref<InputEvent> InputEventMouseButton::xformed_by(const Transform2D &p_xform, const Vector2 &p_local_ofs) const {
	Vector2 g = get_global_position();
	Vector2 l = p_xform.xform(get_position() + p_local_ofs);

	Ref<InputEventMouseButton> mb;
	mb.instantiate();

	mb->set_device(get_device());
	mb->set_window_id(get_window_id());
	mb->set_modifiers_from_event(this);

	mb->set_position(l);
	mb->set_global_position(g);

	mb->set_button_mask(get_button_mask());
	mb->set_pressed(pressed);
	mb->set_double_click(double_click);
	mb->set_factor(factor);
	mb->set_button_index(button_index);

	return mb;
}

bool InputEventMouseButton::action_match(const Ref<InputEvent> &p_event, bool *p_pressed, float *p_strength, float *p_raw_strength, float p_deadzone) const {
	Ref<InputEventMouseButton> mb = p_event;
	if (mb.is_null()) {
		return false;
	}

	bool match = mb->button_index == button_index;
	if (match) {
		bool pressed = mb->is_pressed();
		if (p_pressed != nullptr) {
			*p_pressed = pressed;
		}
		float strength = pressed ? 1.0f : 0.0f;
		if (p_strength != nullptr) {
			*p_strength = strength;
		}
		if (p_raw_strength != nullptr) {
			*p_raw_strength = strength;
		}
	}

	return match;
}

bool InputEventMouseButton::is_match(const Ref<InputEvent> &p_event, bool p_exact_match) const {
	Ref<InputEventMouseButton> mb = p_event;
	if (mb.is_null()) {
		return false;
	}

	return button_index == mb->button_index &&
			(!p_exact_match || get_modifiers_mask() == mb->get_modifiers_mask());
}

static const char *_mouse_button_descriptions[9] = {
	TTRC("Left Mouse Button"),
	TTRC("Right Mouse Button"),
	TTRC("Middle Mouse Button"),
	TTRC("Mouse Wheel Up"),
	TTRC("Mouse Wheel Down"),
	TTRC("Mouse Wheel Left"),
	TTRC("Mouse Wheel Right"),
	TTRC("Mouse Thumb Button 1"),
	TTRC("Mouse Thumb Button 2")
};

String InputEventMouseButton::as_text() const {
	// Modifiers
	String mods_text = InputEventWithModifiers::as_text();
	String full_string = mods_text == "" ? "" : mods_text + "+";

	// Button
	int idx = get_button_index();
	switch (idx) {
		case MOUSE_BUTTON_LEFT:
		case MOUSE_BUTTON_RIGHT:
		case MOUSE_BUTTON_MIDDLE:
		case MOUSE_BUTTON_WHEEL_UP:
		case MOUSE_BUTTON_WHEEL_DOWN:
		case MOUSE_BUTTON_WHEEL_LEFT:
		case MOUSE_BUTTON_WHEEL_RIGHT:
		case MOUSE_BUTTON_XBUTTON1:
		case MOUSE_BUTTON_XBUTTON2:
			full_string += RTR(_mouse_button_descriptions[idx - 1]); // button index starts from 1, array index starts from 0, so subtract 1
			break;
		default:
			full_string += RTR("Button") + " #" + itos(idx);
			break;
	}

	// Double Click
	if (double_click) {
		full_string += " (" + RTR("Double Click") + ")";
	}

	return full_string;
}

String InputEventMouseButton::to_string() {
	String p = is_pressed() ? "true" : "false";
	String d = double_click ? "true" : "false";

	int idx = get_button_index();
	String button_string = itos(idx);

	switch (idx) {
		case MOUSE_BUTTON_LEFT:
		case MOUSE_BUTTON_RIGHT:
		case MOUSE_BUTTON_MIDDLE:
		case MOUSE_BUTTON_WHEEL_UP:
		case MOUSE_BUTTON_WHEEL_DOWN:
		case MOUSE_BUTTON_WHEEL_LEFT:
		case MOUSE_BUTTON_WHEEL_RIGHT:
		case MOUSE_BUTTON_XBUTTON1:
		case MOUSE_BUTTON_XBUTTON2:
			button_string += " (" + RTR(_mouse_button_descriptions[idx - 1]) + ")"; // button index starts from 1, array index starts from 0, so subtract 1
			break;
		default:
			break;
	}

	String mods = InputEventWithModifiers::as_text();
	mods = mods == "" ? TTR("none") : mods;

	// Work around the fact vformat can only take 5 substitutions but 6 need to be passed.
	String index_and_mods = vformat("button_index=%s, mods=%s", button_index, mods);
	return vformat("InputEventMouseButton: %s, pressed=%s, position=(%s), button_mask=%d, double_click=%s", index_and_mods, p, String(get_position()), get_button_mask(), d);
}

void InputEventMouseButton::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_factor", "factor"), &InputEventMouseButton::set_factor);
	ClassDB::bind_method(D_METHOD("get_factor"), &InputEventMouseButton::get_factor);

	ClassDB::bind_method(D_METHOD("set_button_index", "button_index"), &InputEventMouseButton::set_button_index);
	ClassDB::bind_method(D_METHOD("get_button_index"), &InputEventMouseButton::get_button_index);

	ClassDB::bind_method(D_METHOD("set_pressed", "pressed"), &InputEventMouseButton::set_pressed);
	//	ClassDB::bind_method(D_METHOD("is_pressed"), &InputEventMouseButton::is_pressed);

	ClassDB::bind_method(D_METHOD("set_double_click", "double_click"), &InputEventMouseButton::set_double_click);
	ClassDB::bind_method(D_METHOD("is_double_click"), &InputEventMouseButton::is_double_click);

	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "factor"), "set_factor", "get_factor");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "button_index"), "set_button_index", "get_button_index");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "pressed"), "set_pressed", "is_pressed");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "double_click"), "set_double_click", "is_double_click");
}

///////////////////////////////////

void InputEventMouseMotion::set_tilt(const Vector2 &p_tilt) {
	tilt = p_tilt;
}

Vector2 InputEventMouseMotion::get_tilt() const {
	return tilt;
}

void InputEventMouseMotion::set_pressure(float p_pressure) {
	pressure = p_pressure;
}

float InputEventMouseMotion::get_pressure() const {
	return pressure;
}

void InputEventMouseMotion::set_relative(const Vector2 &p_relative) {
	relative = p_relative;
}

Vector2 InputEventMouseMotion::get_relative() const {
	return relative;
}

void InputEventMouseMotion::set_speed(const Vector2 &p_speed) {
	speed = p_speed;
}

Vector2 InputEventMouseMotion::get_speed() const {
	return speed;
}

Ref<InputEvent> InputEventMouseMotion::xformed_by(const Transform2D &p_xform, const Vector2 &p_local_ofs) const {
	Vector2 g = get_global_position();
	Vector2 l = p_xform.xform(get_position() + p_local_ofs);
	Vector2 r = p_xform.basis_xform(get_relative());
	Vector2 s = p_xform.basis_xform(get_speed());

	Ref<InputEventMouseMotion> mm;
	mm.instantiate();

	mm->set_device(get_device());
	mm->set_window_id(get_window_id());

	mm->set_modifiers_from_event(this);

	mm->set_position(l);
	mm->set_pressure(get_pressure());
	mm->set_tilt(get_tilt());
	mm->set_global_position(g);

	mm->set_button_mask(get_button_mask());
	mm->set_relative(r);
	mm->set_speed(s);

	return mm;
}

String InputEventMouseMotion::as_text() const {
	return vformat(RTR("Mouse motion at position (%s) with speed (%s)"), String(get_position()), String(get_speed()));
}

String InputEventMouseMotion::to_string() {
	int button_mask = get_button_mask();
	String button_mask_string = itos(button_mask);
	switch (get_button_mask()) {
		case MOUSE_BUTTON_MASK_LEFT:
			button_mask_string += " (" + RTR(_mouse_button_descriptions[MOUSE_BUTTON_LEFT - 1]) + ")";
			break;
		case MOUSE_BUTTON_MASK_MIDDLE:
			button_mask_string += " (" + RTR(_mouse_button_descriptions[MOUSE_BUTTON_MIDDLE - 1]) + ")";
			break;
		case MOUSE_BUTTON_MASK_RIGHT:
			button_mask_string += " (" + RTR(_mouse_button_descriptions[MOUSE_BUTTON_RIGHT - 1]) + ")";
			break;
		case MOUSE_BUTTON_MASK_XBUTTON1:
			button_mask_string += " (" + RTR(_mouse_button_descriptions[MOUSE_BUTTON_XBUTTON1 - 1]) + ")";
			break;
		case MOUSE_BUTTON_MASK_XBUTTON2:
			button_mask_string += " (" + RTR(_mouse_button_descriptions[MOUSE_BUTTON_XBUTTON2 - 1]) + ")";
			break;
		default:
			break;
	}

	// Work around the fact vformat can only take 5 substitutions but 6 need to be passed.
	String mask_and_position = vformat("button_mask=%s, position=(%s)", button_mask_string, String(get_position()));
	return vformat("InputEventMouseMotion: %s, relative=(%s), speed=(%s), pressure=%.2f, tilt=(%s)", mask_and_position, String(get_relative()), String(get_speed()), get_pressure(), String(get_tilt()));
}

bool InputEventMouseMotion::accumulate(const Ref<InputEvent> &p_event) {
	Ref<InputEventMouseMotion> motion = p_event;
	if (motion.is_null()) {
		return false;
	}

	if (get_window_id() != motion->get_window_id()) {
		return false;
	}

	if (is_pressed() != motion->is_pressed()) {
		return false;
	}

	if (get_button_mask() != motion->get_button_mask()) {
		return false;
	}

	if (is_shift_pressed() != motion->is_shift_pressed()) {
		return false;
	}

	if (is_ctrl_pressed() != motion->is_ctrl_pressed()) {
		return false;
	}

	if (is_alt_pressed() != motion->is_alt_pressed()) {
		return false;
	}

	if (is_meta_pressed() != motion->is_meta_pressed()) {
		return false;
	}

	set_position(motion->get_position());
	set_global_position(motion->get_global_position());
	set_speed(motion->get_speed());
	relative += motion->get_relative();

	return true;
}

void InputEventMouseMotion::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_tilt", "tilt"), &InputEventMouseMotion::set_tilt);
	ClassDB::bind_method(D_METHOD("get_tilt"), &InputEventMouseMotion::get_tilt);

	ClassDB::bind_method(D_METHOD("set_pressure", "pressure"), &InputEventMouseMotion::set_pressure);
	ClassDB::bind_method(D_METHOD("get_pressure"), &InputEventMouseMotion::get_pressure);

	ClassDB::bind_method(D_METHOD("set_relative", "relative"), &InputEventMouseMotion::set_relative);
	ClassDB::bind_method(D_METHOD("get_relative"), &InputEventMouseMotion::get_relative);

	ClassDB::bind_method(D_METHOD("set_speed", "speed"), &InputEventMouseMotion::set_speed);
	ClassDB::bind_method(D_METHOD("get_speed"), &InputEventMouseMotion::get_speed);

	ADD_PROPERTY(PropertyInfo(Variant::VECTOR2, "tilt"), "set_tilt", "get_tilt");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "pressure"), "set_pressure", "get_pressure");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR2, "relative"), "set_relative", "get_relative");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR2, "speed"), "set_speed", "get_speed");
}

///////////////////////////////////

void InputEventJoypadMotion::set_axis(JoyAxis p_axis) {
	ERR_FAIL_INDEX(p_axis, JOY_AXIS_MAX);

	axis = p_axis;
	emit_changed();
}

JoyAxis InputEventJoypadMotion::get_axis() const {
	return axis;
}

void InputEventJoypadMotion::set_axis_value(float p_value) {
	axis_value = p_value;
	emit_changed();
}

float InputEventJoypadMotion::get_axis_value() const {
	return axis_value;
}

bool InputEventJoypadMotion::is_pressed() const {
	return Math::abs(axis_value) >= 0.5f;
}

bool InputEventJoypadMotion::action_match(const Ref<InputEvent> &p_event, bool *p_pressed, float *p_strength, float *p_raw_strength, float p_deadzone) const {
	Ref<InputEventJoypadMotion> jm = p_event;
	if (jm.is_null()) {
		return false;
	}

	bool match = (axis == jm->axis); // Matches even if not in the same direction, but returns a "not pressed" event.
	if (match) {
		float jm_abs_axis_value = Math::abs(jm->get_axis_value());
		bool same_direction = (((axis_value < 0) == (jm->axis_value < 0)) || jm->axis_value == 0);
		bool pressed = same_direction && jm_abs_axis_value >= p_deadzone;
		if (p_pressed != nullptr) {
			*p_pressed = pressed;
		}
		if (p_strength != nullptr) {
			if (pressed) {
				if (p_deadzone == 1.0f) {
					*p_strength = 1.0f;
				} else {
					*p_strength = CLAMP(Math::inverse_lerp(p_deadzone, 1.0f, jm_abs_axis_value), 0.0f, 1.0f);
				}
			} else {
				*p_strength = 0.0f;
			}
		}
		if (p_raw_strength != nullptr) {
			if (same_direction) { // NOT pressed, because we want to ignore the deadzone.
				*p_raw_strength = jm_abs_axis_value;
			} else {
				*p_raw_strength = 0.0f;
			}
		}
	}
	return match;
}

bool InputEventJoypadMotion::is_match(const Ref<InputEvent> &p_event, bool p_exact_match) const {
	Ref<InputEventJoypadMotion> jm = p_event;
	if (jm.is_null()) {
		return false;
	}

	return axis == jm->axis &&
			(!p_exact_match || ((axis_value < 0) == (jm->axis_value < 0)));
}

static const char *_joy_axis_descriptions[JOY_AXIS_MAX] = {
	TTRC("Left Stick X-Axis, Joystick 0 X-Axis"),
	TTRC("Left Stick Y-Axis, Joystick 0 Y-Axis"),
	TTRC("Right Stick X-Axis, Joystick 1 X-Axis"),
	TTRC("Right Stick Y-Axis, Joystick 1 Y-Axis"),
	TTRC("Joystick 2 X-Axis, Left Trigger, Sony L2, Xbox LT"),
	TTRC("Joystick 2 Y-Axis, Right Trigger, Sony R2, Xbox RT"),
	TTRC("Joystick 3 X-Axis"),
	TTRC("Joystick 3 Y-Axis"),
	TTRC("Joystick 4 X-Axis"),
	TTRC("Joystick 4 Y-Axis"),
};

String InputEventJoypadMotion::as_text() const {
	String desc = axis < JOY_AXIS_MAX ? RTR(_joy_axis_descriptions[axis]) : TTR("Unknown Joypad Axis");

	return vformat(TTR("Joypad Motion on Axis %d (%s) with Value %.2f"), axis, desc, axis_value);
}

String InputEventJoypadMotion::to_string() {
	return vformat("InputEventJoypadMotion: axis=%d, axis_value=%.2f", axis, axis_value);
}

void InputEventJoypadMotion::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_axis", "axis"), &InputEventJoypadMotion::set_axis);
	ClassDB::bind_method(D_METHOD("get_axis"), &InputEventJoypadMotion::get_axis);

	ClassDB::bind_method(D_METHOD("set_axis_value", "axis_value"), &InputEventJoypadMotion::set_axis_value);
	ClassDB::bind_method(D_METHOD("get_axis_value"), &InputEventJoypadMotion::get_axis_value);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "axis"), "set_axis", "get_axis");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "axis_value"), "set_axis_value", "get_axis_value");
}

///////////////////////////////////

void InputEventJoypadButton::set_button_index(JoyButton p_index) {
	button_index = p_index;
	emit_changed();
}

JoyButton InputEventJoypadButton::get_button_index() const {
	return button_index;
}

void InputEventJoypadButton::set_pressed(bool p_pressed) {
	pressed = p_pressed;
}

bool InputEventJoypadButton::is_pressed() const {
	return pressed;
}

void InputEventJoypadButton::set_pressure(float p_pressure) {
	pressure = p_pressure;
}

float InputEventJoypadButton::get_pressure() const {
	return pressure;
}

bool InputEventJoypadButton::action_match(const Ref<InputEvent> &p_event, bool *p_pressed, float *p_strength, float *p_raw_strength, float p_deadzone) const {
	Ref<InputEventJoypadButton> jb = p_event;
	if (jb.is_null()) {
		return false;
	}

	bool match = button_index == jb->button_index;
	if (match) {
		bool pressed = jb->is_pressed();
		if (p_pressed != nullptr) {
			*p_pressed = pressed;
		}
		float strength = pressed ? 1.0f : 0.0f;
		if (p_strength != nullptr) {
			*p_strength = strength;
		}
		if (p_raw_strength != nullptr) {
			*p_raw_strength = strength;
		}
	}

	return match;
}

bool InputEventJoypadButton::is_match(const Ref<InputEvent> &p_event, bool p_exact_match) const {
	Ref<InputEventJoypadButton> button = p_event;
	if (button.is_null()) {
		return false;
	}

	return button_index == button->button_index;
}

static const char *_joy_button_descriptions[JOY_BUTTON_SDL_MAX] = {
	TTRC("Bottom Action, Sony Cross, Xbox A, Nintendo B"),
	TTRC("Right Action, Sony Circle, Xbox B, Nintendo A"),
	TTRC("Left Action, Sony Square, Xbox X, Nintendo Y"),
	TTRC("Top Action, Sony Triangle, Xbox Y, Nintendo X"),
	TTRC("Back, Sony Select, Xbox Back, Nintendo -"),
	TTRC("Guide, Sony PS, Xbox Home"),
	TTRC("Start, Nintendo +"),
	TTRC("Left Stick, Sony L3, Xbox L/LS"),
	TTRC("Right Stick, Sony R3, Xbox R/RS"),
	TTRC("Left Shoulder, Sony L1, Xbox LB"),
	TTRC("Right Shoulder, Sony R1, Xbox RB"),
	TTRC("D-pad Up"),
	TTRC("D-pad Down"),
	TTRC("D-pad Left"),
	TTRC("D-pad Right"),
	TTRC("Xbox Share, PS5 Microphone, Nintendo Capture"),
	TTRC("Xbox Paddle 1"),
	TTRC("Xbox Paddle 2"),
	TTRC("Xbox Paddle 3"),
	TTRC("Xbox Paddle 4"),
	TTRC("PS4/5 Touchpad"),
};

String InputEventJoypadButton::as_text() const {
	String text = "Joypad Button " + itos(button_index);

	if (button_index >= 0 && button_index < JOY_BUTTON_SDL_MAX) {
		text += vformat(" (%s)", _joy_button_descriptions[button_index]);
	}

	if (pressure != 0) {
		text += ", Pressure:" + String(Variant(pressure));
	}

	return text;
}

String InputEventJoypadButton::to_string() {
	String p = pressed ? "true" : "false";
	return vformat("InputEventJoypadButton: button_index=%d, pressed=%s, pressure=%.2f", button_index, p, pressure);
}

Ref<InputEventJoypadButton> InputEventJoypadButton::create_reference(JoyButton p_btn_index) {
	Ref<InputEventJoypadButton> ie;
	ie.instantiate();
	ie->set_button_index(p_btn_index);

	return ie;
}

void InputEventJoypadButton::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_button_index", "button_index"), &InputEventJoypadButton::set_button_index);
	ClassDB::bind_method(D_METHOD("get_button_index"), &InputEventJoypadButton::get_button_index);

	ClassDB::bind_method(D_METHOD("set_pressure", "pressure"), &InputEventJoypadButton::set_pressure);
	ClassDB::bind_method(D_METHOD("get_pressure"), &InputEventJoypadButton::get_pressure);

	ClassDB::bind_method(D_METHOD("set_pressed", "pressed"), &InputEventJoypadButton::set_pressed);
	//	ClassDB::bind_method(D_METHOD("is_pressed"), &InputEventJoypadButton::is_pressed);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "button_index"), "set_button_index", "get_button_index");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "pressure"), "set_pressure", "get_pressure");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "pressed"), "set_pressed", "is_pressed");
}

///////////////////////////////////

void InputEventScreenTouch::set_index(int p_index) {
	index = p_index;
}

int InputEventScreenTouch::get_index() const {
	return index;
}

void InputEventScreenTouch::set_position(const Vector2 &p_pos) {
	pos = p_pos;
}

Vector2 InputEventScreenTouch::get_position() const {
	return pos;
}

void InputEventScreenTouch::set_pressed(bool p_pressed) {
	pressed = p_pressed;
}

bool InputEventScreenTouch::is_pressed() const {
	return pressed;
}

Ref<InputEvent> InputEventScreenTouch::xformed_by(const Transform2D &p_xform, const Vector2 &p_local_ofs) const {
	Ref<InputEventScreenTouch> st;
	st.instantiate();
	st->set_device(get_device());
	st->set_window_id(get_window_id());
	st->set_index(index);
	st->set_position(p_xform.xform(pos + p_local_ofs));
	st->set_pressed(pressed);

	return st;
}

String InputEventScreenTouch::as_text() const {
	String status = pressed ? RTR("touched") : RTR("released");

	return vformat(RTR("Screen %s at (%s) with %s touch points"), status, String(get_position()), itos(index));
}

String InputEventScreenTouch::to_string() {
	String p = pressed ? "true" : "false";
	return vformat("InputEventScreenTouch: index=%d, pressed=%s, position=(%s)", index, p, String(get_position()));
}

void InputEventScreenTouch::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_index", "index"), &InputEventScreenTouch::set_index);
	ClassDB::bind_method(D_METHOD("get_index"), &InputEventScreenTouch::get_index);

	ClassDB::bind_method(D_METHOD("set_position", "position"), &InputEventScreenTouch::set_position);
	ClassDB::bind_method(D_METHOD("get_position"), &InputEventScreenTouch::get_position);

	ClassDB::bind_method(D_METHOD("set_pressed", "pressed"), &InputEventScreenTouch::set_pressed);
	//ClassDB::bind_method(D_METHOD("is_pressed"),&InputEventScreenTouch::is_pressed);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "index"), "set_index", "get_index");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR2, "position"), "set_position", "get_position");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "pressed"), "set_pressed", "is_pressed");
}

///////////////////////////////////

void InputEventScreenDrag::set_index(int p_index) {
	index = p_index;
}

int InputEventScreenDrag::get_index() const {
	return index;
}

void InputEventScreenDrag::set_position(const Vector2 &p_pos) {
	pos = p_pos;
}

Vector2 InputEventScreenDrag::get_position() const {
	return pos;
}

void InputEventScreenDrag::set_relative(const Vector2 &p_relative) {
	relative = p_relative;
}

Vector2 InputEventScreenDrag::get_relative() const {
	return relative;
}

void InputEventScreenDrag::set_speed(const Vector2 &p_speed) {
	speed = p_speed;
}

Vector2 InputEventScreenDrag::get_speed() const {
	return speed;
}

Ref<InputEvent> InputEventScreenDrag::xformed_by(const Transform2D &p_xform, const Vector2 &p_local_ofs) const {
	Ref<InputEventScreenDrag> sd;

	sd.instantiate();

	sd->set_device(get_device());
	sd->set_window_id(get_window_id());

	sd->set_index(index);
	sd->set_position(p_xform.xform(pos + p_local_ofs));
	sd->set_relative(p_xform.basis_xform(relative));
	sd->set_speed(p_xform.basis_xform(speed));

	return sd;
}

String InputEventScreenDrag::as_text() const {
	return vformat(RTR("Screen dragged with %s touch points at position (%s) with speed of (%s)"), itos(index), String(get_position()), String(get_speed()));
}

String InputEventScreenDrag::to_string() {
	return vformat("InputEventScreenDrag: index=%d, position=(%s), relative=(%s), speed=(%s)", index, String(get_position()), String(get_relative()), String(get_speed()));
}

bool InputEventScreenDrag::accumulate(const Ref<InputEvent> &p_event) {
	Ref<InputEventScreenDrag> drag = p_event;
	if (drag.is_null())
		return false;

	if (get_index() != drag->get_index()) {
		return false;
	}

	set_position(drag->get_position());
	set_speed(drag->get_speed());
	relative += drag->get_relative();

	return true;
}

void InputEventScreenDrag::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_index", "index"), &InputEventScreenDrag::set_index);
	ClassDB::bind_method(D_METHOD("get_index"), &InputEventScreenDrag::get_index);

	ClassDB::bind_method(D_METHOD("set_position", "position"), &InputEventScreenDrag::set_position);
	ClassDB::bind_method(D_METHOD("get_position"), &InputEventScreenDrag::get_position);

	ClassDB::bind_method(D_METHOD("set_relative", "relative"), &InputEventScreenDrag::set_relative);
	ClassDB::bind_method(D_METHOD("get_relative"), &InputEventScreenDrag::get_relative);

	ClassDB::bind_method(D_METHOD("set_speed", "speed"), &InputEventScreenDrag::set_speed);
	ClassDB::bind_method(D_METHOD("get_speed"), &InputEventScreenDrag::get_speed);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "index"), "set_index", "get_index");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR2, "position"), "set_position", "get_position");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR2, "relative"), "set_relative", "get_relative");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR2, "speed"), "set_speed", "get_speed");
}

///////////////////////////////////

void InputEventAction::set_action(const StringName &p_action) {
	action = p_action;
}

StringName InputEventAction::get_action() const {
	return action;
}

void InputEventAction::set_pressed(bool p_pressed) {
	pressed = p_pressed;
}

bool InputEventAction::is_pressed() const {
	return pressed;
}

void InputEventAction::set_strength(float p_strength) {
	strength = CLAMP(p_strength, 0.0f, 1.0f);
}

float InputEventAction::get_strength() const {
	return strength;
}

bool InputEventAction::is_match(const Ref<InputEvent> &p_event, bool p_exact_match) const {
	if (p_event.is_null()) {
		return false;
	}

	return p_event->is_action(action);
}

bool InputEventAction::is_action(const StringName &p_action) const {
	return action == p_action;
}

bool InputEventAction::action_match(const Ref<InputEvent> &p_event, bool *p_pressed, float *p_strength, float *p_raw_strength, float p_deadzone) const {
	Ref<InputEventAction> act = p_event;
	if (act.is_null()) {
		return false;
	}

	bool match = action == act->action;
	if (match) {
		bool pressed = act->pressed;
		if (p_pressed != nullptr) {
			*p_pressed = pressed;
		}
		float strength = pressed ? 1.0f : 0.0f;
		if (p_strength != nullptr) {
			*p_strength = strength;
		}
		if (p_raw_strength != nullptr) {
			*p_raw_strength = strength;
		}
	}
	return match;
}

String InputEventAction::as_text() const {
	return vformat(RTR("Input Action %s was %s"), action, pressed ? "pressed" : "released");
}

String InputEventAction::to_string() {
	String p = pressed ? "true" : "false";
	return vformat("InputEventAction: action=\"%s\", pressed=%s", action, p);
}

void InputEventAction::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_action", "action"), &InputEventAction::set_action);
	ClassDB::bind_method(D_METHOD("get_action"), &InputEventAction::get_action);

	ClassDB::bind_method(D_METHOD("set_pressed", "pressed"), &InputEventAction::set_pressed);
	//ClassDB::bind_method(D_METHOD("is_pressed"), &InputEventAction::is_pressed);

	ClassDB::bind_method(D_METHOD("set_strength", "strength"), &InputEventAction::set_strength);
	ClassDB::bind_method(D_METHOD("get_strength"), &InputEventAction::get_strength);

	//	ClassDB::bind_method(D_METHOD("is_action", "name"), &InputEventAction::is_action);

	ADD_PROPERTY(PropertyInfo(Variant::STRING_NAME, "action"), "set_action", "get_action");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "pressed"), "set_pressed", "is_pressed");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "strength", PROPERTY_HINT_RANGE, "0,1,0.01"), "set_strength", "get_strength");
}

///////////////////////////////////

void InputEventGesture::set_position(const Vector2 &p_pos) {
	pos = p_pos;
}

void InputEventGesture::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_position", "position"), &InputEventGesture::set_position);
	ClassDB::bind_method(D_METHOD("get_position"), &InputEventGesture::get_position);

	ADD_PROPERTY(PropertyInfo(Variant::VECTOR2, "position"), "set_position", "get_position");
}

Vector2 InputEventGesture::get_position() const {
	return pos;
}

///////////////////////////////////

void InputEventMagnifyGesture::set_factor(real_t p_factor) {
	factor = p_factor;
}

real_t InputEventMagnifyGesture::get_factor() const {
	return factor;
}

Ref<InputEvent> InputEventMagnifyGesture::xformed_by(const Transform2D &p_xform, const Vector2 &p_local_ofs) const {
	Ref<InputEventMagnifyGesture> ev;
	ev.instantiate();

	ev->set_device(get_device());
	ev->set_window_id(get_window_id());

	ev->set_modifiers_from_event(this);

	ev->set_position(p_xform.xform(get_position() + p_local_ofs));
	ev->set_factor(get_factor());

	return ev;
}

String InputEventMagnifyGesture::as_text() const {
	return vformat(RTR("Magnify Gesture at (%s) with factor %s"), String(get_position()), rtos(get_factor()));
}

String InputEventMagnifyGesture::to_string() {
	return vformat("InputEventMagnifyGesture: factor=%.2f, position=(%s)", factor, String(get_position()));
}

void InputEventMagnifyGesture::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_factor", "factor"), &InputEventMagnifyGesture::set_factor);
	ClassDB::bind_method(D_METHOD("get_factor"), &InputEventMagnifyGesture::get_factor);

	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "factor"), "set_factor", "get_factor");
}

///////////////////////////////////

void InputEventPanGesture::set_delta(const Vector2 &p_delta) {
	delta = p_delta;
}

Vector2 InputEventPanGesture::get_delta() const {
	return delta;
}

Ref<InputEvent> InputEventPanGesture::xformed_by(const Transform2D &p_xform, const Vector2 &p_local_ofs) const {
	Ref<InputEventPanGesture> ev;
	ev.instantiate();

	ev->set_device(get_device());
	ev->set_window_id(get_window_id());

	ev->set_modifiers_from_event(this);

	ev->set_position(p_xform.xform(get_position() + p_local_ofs));
	ev->set_delta(get_delta());

	return ev;
}

String InputEventPanGesture::as_text() const {
	return vformat(RTR("Pan Gesture at (%s) with delta (%s)"), String(get_position()), String(get_delta()));
}

String InputEventPanGesture::to_string() {
	return vformat("InputEventPanGesture: delta=(%s), position=(%s)", String(get_delta()), String(get_position()));
}

void InputEventPanGesture::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_delta", "delta"), &InputEventPanGesture::set_delta);
	ClassDB::bind_method(D_METHOD("get_delta"), &InputEventPanGesture::get_delta);

	ADD_PROPERTY(PropertyInfo(Variant::VECTOR2, "delta"), "set_delta", "get_delta");
}

///////////////////////////////////

void InputEventMIDI::set_channel(const int p_channel) {
	channel = p_channel;
}

int InputEventMIDI::get_channel() const {
	return channel;
}

void InputEventMIDI::set_message(const MIDIMessage p_message) {
	message = p_message;
}

MIDIMessage InputEventMIDI::get_message() const {
	return message;
}

void InputEventMIDI::set_pitch(const int p_pitch) {
	pitch = p_pitch;
}

int InputEventMIDI::get_pitch() const {
	return pitch;
}

void InputEventMIDI::set_velocity(const int p_velocity) {
	velocity = p_velocity;
}

int InputEventMIDI::get_velocity() const {
	return velocity;
}

void InputEventMIDI::set_instrument(const int p_instrument) {
	instrument = p_instrument;
}

int InputEventMIDI::get_instrument() const {
	return instrument;
}

void InputEventMIDI::set_pressure(const int p_pressure) {
	pressure = p_pressure;
}

int InputEventMIDI::get_pressure() const {
	return pressure;
}

void InputEventMIDI::set_controller_number(const int p_controller_number) {
	controller_number = p_controller_number;
}

int InputEventMIDI::get_controller_number() const {
	return controller_number;
}

void InputEventMIDI::set_controller_value(const int p_controller_value) {
	controller_value = p_controller_value;
}

int InputEventMIDI::get_controller_value() const {
	return controller_value;
}

String InputEventMIDI::as_text() const {
	return vformat(RTR("MIDI Input on Channel=%s Message=%s"), itos(channel), itos(message));
}

String InputEventMIDI::to_string() {
	return vformat("InputEventMIDI: channel=%d, message=%d, pitch=%d, velocity=%d, pressure=%d", channel, message, pitch, velocity, pressure);
}

void InputEventMIDI::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_channel", "channel"), &InputEventMIDI::set_channel);
	ClassDB::bind_method(D_METHOD("get_channel"), &InputEventMIDI::get_channel);
	ClassDB::bind_method(D_METHOD("set_message", "message"), &InputEventMIDI::set_message);
	ClassDB::bind_method(D_METHOD("get_message"), &InputEventMIDI::get_message);
	ClassDB::bind_method(D_METHOD("set_pitch", "pitch"), &InputEventMIDI::set_pitch);
	ClassDB::bind_method(D_METHOD("get_pitch"), &InputEventMIDI::get_pitch);
	ClassDB::bind_method(D_METHOD("set_velocity", "velocity"), &InputEventMIDI::set_velocity);
	ClassDB::bind_method(D_METHOD("get_velocity"), &InputEventMIDI::get_velocity);
	ClassDB::bind_method(D_METHOD("set_instrument", "instrument"), &InputEventMIDI::set_instrument);
	ClassDB::bind_method(D_METHOD("get_instrument"), &InputEventMIDI::get_instrument);
	ClassDB::bind_method(D_METHOD("set_pressure", "pressure"), &InputEventMIDI::set_pressure);
	ClassDB::bind_method(D_METHOD("get_pressure"), &InputEventMIDI::get_pressure);
	ClassDB::bind_method(D_METHOD("set_controller_number", "controller_number"), &InputEventMIDI::set_controller_number);
	ClassDB::bind_method(D_METHOD("get_controller_number"), &InputEventMIDI::get_controller_number);
	ClassDB::bind_method(D_METHOD("set_controller_value", "controller_value"), &InputEventMIDI::set_controller_value);
	ClassDB::bind_method(D_METHOD("get_controller_value"), &InputEventMIDI::get_controller_value);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "channel"), "set_channel", "get_channel");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "message"), "set_message", "get_message");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "pitch"), "set_pitch", "get_pitch");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "velocity"), "set_velocity", "get_velocity");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "instrument"), "set_instrument", "get_instrument");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "pressure"), "set_pressure", "get_pressure");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "controller_number"), "set_controller_number", "get_controller_number");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "controller_value"), "set_controller_value", "get_controller_value");
}

///////////////////////////////////

void InputEventShortcut::set_shortcut(Ref<Shortcut> p_shortcut) {
	shortcut = p_shortcut;
	emit_changed();
}

Ref<Shortcut> InputEventShortcut::get_shortcut() {
	return shortcut;
}

void InputEventShortcut::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_shortcut", "shortcut"), &InputEventShortcut::set_shortcut);
	ClassDB::bind_method(D_METHOD("get_shortcut"), &InputEventShortcut::get_shortcut);

	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "shortcut", PROPERTY_HINT_RESOURCE_TYPE, "Shortcut"), "set_shortcut", "get_shortcut");
}

bool InputEventShortcut::is_pressed() const {
	return true;
}

String InputEventShortcut::as_text() const {
	ERR_FAIL_COND_V(shortcut.is_null(), "None");

	return vformat(RTR("Input Event with Shortcut=%s"), shortcut->get_as_text());
}

String InputEventShortcut::to_string() {
	ERR_FAIL_COND_V(shortcut.is_null(), "None");

	return vformat("InputEventShortcut: shortcut=%s", shortcut->get_as_text());
}
