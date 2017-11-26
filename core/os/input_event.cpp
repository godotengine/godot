/*************************************************************************/
/*  input_event.cpp                                                      */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
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

void InputEvent::set_id(uint32_t p_id) {
	id = p_id;
}

uint32_t InputEvent::get_id() const {
	return id;
}

void InputEvent::set_device(int p_device) {
	device = p_device;
}

int InputEvent::get_device() const {
	return device;
}

bool InputEvent::is_pressed() const {

	return false;
}

bool InputEvent::is_action(const StringName &p_action) const {

	return InputMap::get_singleton()->event_is_action(Ref<InputEvent>((InputEvent *)this), p_action);
}

bool InputEvent::is_action_pressed(const StringName &p_action) const {

	return (is_pressed() && !is_echo() && is_action(p_action));
}
bool InputEvent::is_action_released(const StringName &p_action) const {

	return (!is_pressed() && is_action(p_action));
}

bool InputEvent::is_echo() const {

	return false;
}

Ref<InputEvent> InputEvent::xformed_by(const Transform2D &p_xform, const Vector2 &p_local_ofs) const {

	return Ref<InputEvent>((InputEvent *)this);
}

String InputEvent::as_text() const {

	return String();
}

bool InputEvent::action_match(const Ref<InputEvent> &p_event) const {

	return false;
}

bool InputEvent::shortcut_match(const Ref<InputEvent> &p_event) const {

	return false;
}

bool InputEvent::is_action_type() const {

	return false;
}

void InputEvent::_bind_methods() {

	ClassDB::bind_method(D_METHOD("set_id", "id"), &InputEvent::set_id);
	ClassDB::bind_method(D_METHOD("get_id"), &InputEvent::get_id);

	ClassDB::bind_method(D_METHOD("set_device", "device"), &InputEvent::set_device);
	ClassDB::bind_method(D_METHOD("get_device"), &InputEvent::get_device);

	ClassDB::bind_method(D_METHOD("is_pressed"), &InputEvent::is_pressed);
	ClassDB::bind_method(D_METHOD("is_action", "action"), &InputEvent::is_action);
	ClassDB::bind_method(D_METHOD("is_action_pressed", "action"), &InputEvent::is_action_pressed);
	ClassDB::bind_method(D_METHOD("is_action_released", "action"), &InputEvent::is_action_released);
	ClassDB::bind_method(D_METHOD("is_echo"), &InputEvent::is_echo);

	ClassDB::bind_method(D_METHOD("as_text"), &InputEvent::as_text);

	ClassDB::bind_method(D_METHOD("action_match", "event"), &InputEvent::action_match);
	ClassDB::bind_method(D_METHOD("shortcut_match", "event"), &InputEvent::shortcut_match);

	ClassDB::bind_method(D_METHOD("is_action_type"), &InputEvent::is_action_type);

	ClassDB::bind_method(D_METHOD("xformed_by", "xform", "local_ofs"), &InputEvent::xformed_by, DEFVAL(Vector2()));

	ADD_PROPERTY(PropertyInfo(Variant::INT, "device"), "set_device", "get_device");
}

InputEvent::InputEvent() {

	id = 0;
	device = 0;
}

//////////////////

void InputEventWithModifiers::set_shift(bool p_enabled) {

	shift = p_enabled;
}

bool InputEventWithModifiers::get_shift() const {

	return shift;
}

void InputEventWithModifiers::set_alt(bool p_enabled) {

	alt = p_enabled;
}
bool InputEventWithModifiers::get_alt() const {

	return alt;
}

void InputEventWithModifiers::set_control(bool p_enabled) {

	control = p_enabled;
}
bool InputEventWithModifiers::get_control() const {

	return control;
}

void InputEventWithModifiers::set_metakey(bool p_enabled) {

	meta = p_enabled;
}
bool InputEventWithModifiers::get_metakey() const {

	return meta;
}

void InputEventWithModifiers::set_command(bool p_enabled) {

	command = p_enabled;
}
bool InputEventWithModifiers::get_command() const {

	return command;
}

void InputEventWithModifiers::set_modifiers_from_event(const InputEventWithModifiers *event) {

	set_alt(event->get_alt());
	set_shift(event->get_shift());
	set_control(event->get_control());
	set_metakey(event->get_metakey());
}

void InputEventWithModifiers::_bind_methods() {

	ClassDB::bind_method(D_METHOD("set_alt", "enable"), &InputEventWithModifiers::set_alt);
	ClassDB::bind_method(D_METHOD("get_alt"), &InputEventWithModifiers::get_alt);

	ClassDB::bind_method(D_METHOD("set_shift", "enable"), &InputEventWithModifiers::set_shift);
	ClassDB::bind_method(D_METHOD("get_shift"), &InputEventWithModifiers::get_shift);

	ClassDB::bind_method(D_METHOD("set_control", "enable"), &InputEventWithModifiers::set_control);
	ClassDB::bind_method(D_METHOD("get_control"), &InputEventWithModifiers::get_control);

	ClassDB::bind_method(D_METHOD("set_metakey", "enable"), &InputEventWithModifiers::set_metakey);
	ClassDB::bind_method(D_METHOD("get_metakey"), &InputEventWithModifiers::get_metakey);

	ClassDB::bind_method(D_METHOD("set_command", "enable"), &InputEventWithModifiers::set_command);
	ClassDB::bind_method(D_METHOD("get_command"), &InputEventWithModifiers::get_command);

	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "alt"), "set_alt", "get_alt");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "shift"), "set_shift", "get_shift");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "control"), "set_control", "get_control");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "meta"), "set_metakey", "get_metakey");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "command"), "set_command", "get_command");
}

InputEventWithModifiers::InputEventWithModifiers() {

	alt = false;
	shift = false;
	control = false;
	meta = false;
}

//////////////////////////////////

void InputEventKey::set_pressed(bool p_pressed) {

	pressed = p_pressed;
}

bool InputEventKey::is_pressed() const {

	return pressed;
}

void InputEventKey::set_scancode(uint32_t p_scancode) {

	scancode = p_scancode;
}
uint32_t InputEventKey::get_scancode() const {

	return scancode;
}

void InputEventKey::set_unicode(uint32_t p_unicode) {

	unicode = p_unicode;
}
uint32_t InputEventKey::get_unicode() const {

	return unicode;
}

void InputEventKey::set_echo(bool p_enable) {

	echo = p_enable;
}
bool InputEventKey::is_echo() const {

	return echo;
}

uint32_t InputEventKey::get_scancode_with_modifiers() const {

	uint32_t sc = scancode;
	if (get_control())
		sc |= KEY_MASK_CTRL;
	if (get_alt())
		sc |= KEY_MASK_ALT;
	if (get_shift())
		sc |= KEY_MASK_SHIFT;
	if (get_metakey())
		sc |= KEY_MASK_META;

	return sc;
}

String InputEventKey::as_text() const {

	String kc = keycode_get_string(scancode);
	if (kc == String())
		return kc;

	if (get_metakey()) {
		kc = find_keycode_name(KEY_META) + ("+" + kc);
	}
	if (get_alt()) {
		kc = find_keycode_name(KEY_ALT) + ("+" + kc);
	}
	if (get_shift()) {
		kc = find_keycode_name(KEY_SHIFT) + ("+" + kc);
	}
	if (get_control()) {
		kc = find_keycode_name(KEY_CONTROL) + ("+" + kc);
	}
	return kc;
}

bool InputEventKey::action_match(const Ref<InputEvent> &p_event) const {

	Ref<InputEventKey> key = p_event;
	if (key.is_null())
		return false;

	uint32_t code = get_scancode_with_modifiers();
	uint32_t event_code = key->get_scancode_with_modifiers();

	return get_scancode() == key->get_scancode() && (!key->is_pressed() || (code & event_code) == code);
}

bool InputEventKey::shortcut_match(const Ref<InputEvent> &p_event) const {

	Ref<InputEventKey> key = p_event;
	if (key.is_null())
		return false;

	uint32_t code = get_scancode_with_modifiers();
	uint32_t event_code = key->get_scancode_with_modifiers();

	return code == event_code;
}

void InputEventKey::_bind_methods() {

	ClassDB::bind_method(D_METHOD("set_pressed", "pressed"), &InputEventKey::set_pressed);

	ClassDB::bind_method(D_METHOD("set_scancode", "scancode"), &InputEventKey::set_scancode);
	ClassDB::bind_method(D_METHOD("get_scancode"), &InputEventKey::get_scancode);

	ClassDB::bind_method(D_METHOD("set_unicode", "unicode"), &InputEventKey::set_unicode);
	ClassDB::bind_method(D_METHOD("get_unicode"), &InputEventKey::get_unicode);

	ClassDB::bind_method(D_METHOD("set_echo", "echo"), &InputEventKey::set_echo);

	ClassDB::bind_method(D_METHOD("get_scancode_with_modifiers"), &InputEventKey::get_scancode_with_modifiers);

	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "pressed"), "set_pressed", "is_pressed");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "scancode"), "set_scancode", "get_scancode");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "unicode"), "set_unicode", "get_unicode");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "echo"), "set_echo", "is_echo");
}

InputEventKey::InputEventKey() {

	pressed = false;
	scancode = 0;
	unicode = 0; ///unicode
	echo = false;
}

////////////////////////////////////////

void InputEventMouse::set_button_mask(int p_mask) {

	button_mask = p_mask;
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

InputEventMouse::InputEventMouse() {

	button_mask = 0;
}

///////////////////////////////////////

void InputEventMouseButton::set_factor(float p_factor) {

	factor = p_factor;
}

float InputEventMouseButton::get_factor() {

	return factor;
}

void InputEventMouseButton::set_button_index(int p_index) {

	button_index = p_index;
}
int InputEventMouseButton::get_button_index() const {

	return button_index;
}

void InputEventMouseButton::set_pressed(bool p_pressed) {

	pressed = p_pressed;
}
bool InputEventMouseButton::is_pressed() const {

	return pressed;
}

void InputEventMouseButton::set_doubleclick(bool p_doubleclick) {

	doubleclick = p_doubleclick;
}
bool InputEventMouseButton::is_doubleclick() const {

	return doubleclick;
}

Ref<InputEvent> InputEventMouseButton::xformed_by(const Transform2D &p_xform, const Vector2 &p_local_ofs) const {

	Vector2 g = p_xform.xform(get_global_position());
	Vector2 l = p_xform.xform(get_position() + p_local_ofs);

	Ref<InputEventMouseButton> mb;
	mb.instance();

	mb->set_id(get_id());
	mb->set_device(get_device());

	mb->set_modifiers_from_event(this);

	mb->set_position(l);
	mb->set_global_position(g);

	mb->set_button_mask(get_button_mask());
	mb->set_pressed(pressed);
	mb->set_doubleclick(doubleclick);
	mb->set_factor(factor);
	mb->set_button_index(button_index);

	return mb;
}

bool InputEventMouseButton::action_match(const Ref<InputEvent> &p_event) const {

	Ref<InputEventMouseButton> mb = p_event;
	if (mb.is_null())
		return false;

	return mb->button_index == button_index;
}

String InputEventMouseButton::as_text() const {

	String button_index_string = "";
	switch (get_button_index()) {
		case BUTTON_LEFT:
			button_index_string = "BUTTON_LEFT";
			break;
		case BUTTON_RIGHT:
			button_index_string = "BUTTON_RIGHT";
			break;
		case BUTTON_MIDDLE:
			button_index_string = "BUTTON_MIDDLE";
			break;
		case BUTTON_WHEEL_UP:
			button_index_string = "BUTTON_WHEEL_UP";
			break;
		case BUTTON_WHEEL_DOWN:
			button_index_string = "BUTTON_WHEEL_DOWN";
			break;
		case BUTTON_WHEEL_LEFT:
			button_index_string = "BUTTON_WHEEL_LEFT";
			break;
		case BUTTON_WHEEL_RIGHT:
			button_index_string = "BUTTON_WHEEL_RIGHT";
			break;
		default:
			button_index_string = itos(get_button_index());
			break;
	}
	return "InputEventMouseButton : button_index=" + button_index_string + ", pressed=" + (pressed ? "true" : "false") + ", position=(" + String(get_position()) + "), button_mask=" + itos(get_button_mask()) + ", doubleclick=" + (doubleclick ? "true" : "false");
}

void InputEventMouseButton::_bind_methods() {

	ClassDB::bind_method(D_METHOD("set_factor", "factor"), &InputEventMouseButton::set_factor);
	ClassDB::bind_method(D_METHOD("get_factor"), &InputEventMouseButton::get_factor);

	ClassDB::bind_method(D_METHOD("set_button_index", "button_index"), &InputEventMouseButton::set_button_index);
	ClassDB::bind_method(D_METHOD("get_button_index"), &InputEventMouseButton::get_button_index);

	ClassDB::bind_method(D_METHOD("set_pressed", "pressed"), &InputEventMouseButton::set_pressed);
	//	ClassDB::bind_method(D_METHOD("is_pressed"), &InputEventMouseButton::is_pressed);

	ClassDB::bind_method(D_METHOD("set_doubleclick", "doubleclick"), &InputEventMouseButton::set_doubleclick);
	ClassDB::bind_method(D_METHOD("is_doubleclick"), &InputEventMouseButton::is_doubleclick);

	ADD_PROPERTY(PropertyInfo(Variant::REAL, "factor"), "set_factor", "get_factor");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "button_index"), "set_button_index", "get_button_index");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "pressed"), "set_pressed", "is_pressed");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "doubleclick"), "set_doubleclick", "is_doubleclick");
}

InputEventMouseButton::InputEventMouseButton() {

	factor = 1;
	button_index = 0;
	pressed = false;
	doubleclick = false;
}

////////////////////////////////////////////

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

	Vector2 g = p_xform.xform(get_global_position());
	Vector2 l = p_xform.xform(get_position() + p_local_ofs);
	Vector2 r = p_xform.basis_xform(get_relative());
	Vector2 s = p_xform.basis_xform(get_speed());

	Ref<InputEventMouseMotion> mm;
	mm.instance();

	mm->set_id(get_id());
	mm->set_device(get_device());

	mm->set_modifiers_from_event(this);

	mm->set_position(l);
	mm->set_global_position(g);

	mm->set_button_mask(get_button_mask());
	mm->set_relative(r);
	mm->set_speed(s);

	return mm;
}

String InputEventMouseMotion::as_text() const {

	String button_mask_string = "";
	switch (get_button_mask()) {
		case BUTTON_MASK_LEFT:
			button_mask_string = "BUTTON_MASK_LEFT";
			break;
		case BUTTON_MASK_MIDDLE:
			button_mask_string = "BUTTON_MASK_MIDDLE";
			break;
		case BUTTON_MASK_RIGHT:
			button_mask_string = "BUTTON_MASK_RIGHT";
			break;
		default:
			button_mask_string = itos(get_button_mask());
			break;
	}
	return "InputEventMouseMotion : button_mask=" + button_mask_string + ", position=(" + String(get_position()) + "), relative=(" + String(get_relative()) + "), speed=(" + String(get_speed()) + ")";
}

void InputEventMouseMotion::_bind_methods() {

	ClassDB::bind_method(D_METHOD("set_relative", "relative"), &InputEventMouseMotion::set_relative);
	ClassDB::bind_method(D_METHOD("get_relative"), &InputEventMouseMotion::get_relative);

	ClassDB::bind_method(D_METHOD("set_speed", "speed"), &InputEventMouseMotion::set_speed);
	ClassDB::bind_method(D_METHOD("get_speed"), &InputEventMouseMotion::get_speed);

	ADD_PROPERTY(PropertyInfo(Variant::VECTOR2, "relative"), "set_relative", "get_relative");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR2, "speed"), "set_speed", "get_speed");
}

InputEventMouseMotion::InputEventMouseMotion() {
}

////////////////////////////////////////

void InputEventJoypadMotion::set_axis(int p_axis) {

	axis = p_axis;
}

int InputEventJoypadMotion::get_axis() const {

	return axis;
}

void InputEventJoypadMotion::set_axis_value(float p_value) {

	axis_value = p_value;
}
float InputEventJoypadMotion::get_axis_value() const {

	return axis_value;
}

bool InputEventJoypadMotion::is_pressed() const {

	return Math::abs(axis_value) > 0.5f;
}

bool InputEventJoypadMotion::action_match(const Ref<InputEvent> &p_event) const {

	Ref<InputEventJoypadMotion> jm = p_event;
	if (jm.is_null())
		return false;

	return (axis == jm->axis && ((axis_value < 0) == (jm->axis_value < 0) || jm->axis_value == 0));
}

String InputEventJoypadMotion::as_text() const {

	return "InputEventJoypadMotion : axis=" + itos(axis) + ", axis_value=" + String(Variant(axis_value));
}

void InputEventJoypadMotion::_bind_methods() {

	ClassDB::bind_method(D_METHOD("set_axis", "axis"), &InputEventJoypadMotion::set_axis);
	ClassDB::bind_method(D_METHOD("get_axis"), &InputEventJoypadMotion::get_axis);

	ClassDB::bind_method(D_METHOD("set_axis_value", "axis_value"), &InputEventJoypadMotion::set_axis_value);
	ClassDB::bind_method(D_METHOD("get_axis_value"), &InputEventJoypadMotion::get_axis_value);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "axis"), "set_axis", "get_axis");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "axis_value"), "set_axis_value", "get_axis_value");
}

InputEventJoypadMotion::InputEventJoypadMotion() {

	axis = 0;
	axis_value = 0;
}
/////////////////////////////////

void InputEventJoypadButton::set_button_index(int p_index) {

	button_index = p_index;
}

int InputEventJoypadButton::get_button_index() const {

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

bool InputEventJoypadButton::action_match(const Ref<InputEvent> &p_event) const {

	Ref<InputEventJoypadButton> jb = p_event;
	if (jb.is_null())
		return false;

	return button_index == jb->button_index;
}

String InputEventJoypadButton::as_text() const {

	return "InputEventJoypadButton : button_index=" + itos(button_index) + ", pressed=" + (pressed ? "true" : "false") + ", pressure=" + String(Variant(pressure));
}

void InputEventJoypadButton::_bind_methods() {

	ClassDB::bind_method(D_METHOD("set_button_index", "button_index"), &InputEventJoypadButton::set_button_index);
	ClassDB::bind_method(D_METHOD("get_button_index"), &InputEventJoypadButton::get_button_index);

	ClassDB::bind_method(D_METHOD("set_pressure", "pressure"), &InputEventJoypadButton::set_pressure);
	ClassDB::bind_method(D_METHOD("get_pressure"), &InputEventJoypadButton::get_pressure);

	ClassDB::bind_method(D_METHOD("set_pressed", "pressed"), &InputEventJoypadButton::set_pressed);
	//	ClassDB::bind_method(D_METHOD("is_pressed"), &InputEventJoypadButton::is_pressed);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "button_index"), "set_button_index", "get_button_index");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "pressure"), "set_pressure", "get_pressure");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "pressed"), "set_pressed", "is_pressed");
}

InputEventJoypadButton::InputEventJoypadButton() {

	button_index = 0;
	pressure = 0;
	pressed = false;
}

//////////////////////////////////////////////

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
	st.instance();
	st->set_id(get_id());
	st->set_device(get_device());
	st->set_index(index);
	st->set_position(p_xform.xform(pos + p_local_ofs));
	st->set_pressed(pressed);

	return st;
}

String InputEventScreenTouch::as_text() const {

	return "InputEventScreenTouch : index=" + itos(index) + ", pressed=" + (pressed ? "true" : "false") + ", position=(" + String(get_position()) + ")";
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

InputEventScreenTouch::InputEventScreenTouch() {

	index = 0;
	pressed = false;
}

/////////////////////////////

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

	sd.instance();

	sd->set_id(get_id());
	sd->set_device(get_device());

	sd->set_index(index);
	sd->set_position(p_xform.xform(pos + p_local_ofs));
	sd->set_relative(p_xform.basis_xform(relative));
	sd->set_speed(p_xform.basis_xform(speed));

	return sd;
}

String InputEventScreenDrag::as_text() const {

	return "InputEventScreenDrag : index=" + itos(index) + ", position=(" + String(get_position()) + "), relative=(" + String(get_relative()) + "), speed=(" + String(get_speed()) + ")";
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

InputEventScreenDrag::InputEventScreenDrag() {

	index = 0;
}
/////////////////////////////

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

bool InputEventAction::is_action(const StringName &p_action) const {

	return action == p_action;
}

String InputEventAction::as_text() const {

	return "InputEventAction : action=" + action + ", pressed=(" + (pressed ? "true" : "false");
}

void InputEventAction::_bind_methods() {

	ClassDB::bind_method(D_METHOD("set_action", "action"), &InputEventAction::set_action);
	ClassDB::bind_method(D_METHOD("get_action"), &InputEventAction::get_action);

	ClassDB::bind_method(D_METHOD("set_pressed", "pressed"), &InputEventAction::set_pressed);
	//ClassDB::bind_method(D_METHOD("is_pressed"), &InputEventAction::is_pressed);

	//	ClassDB::bind_method(D_METHOD("is_action", "name"), &InputEventAction::is_action);

	ADD_PROPERTY(PropertyInfo(Variant::STRING, "action"), "set_action", "get_action");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "pressed"), "set_pressed", "is_pressed");
}

InputEventAction::InputEventAction() {
	pressed = false;
}
/////////////////////////////

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
/////////////////////////////

void InputEventMagnifyGesture::set_factor(real_t p_factor) {

	factor = p_factor;
}

real_t InputEventMagnifyGesture::get_factor() const {

	return factor;
}

Ref<InputEvent> InputEventMagnifyGesture::xformed_by(const Transform2D &p_xform, const Vector2 &p_local_ofs) const {

	Ref<InputEventMagnifyGesture> ev;
	ev.instance();

	ev->set_id(get_id());
	ev->set_device(get_device());
	ev->set_modifiers_from_event(this);

	ev->set_position(p_xform.xform(get_position() + p_local_ofs));
	ev->set_factor(get_factor());

	return ev;
}

void InputEventMagnifyGesture::_bind_methods() {

	ClassDB::bind_method(D_METHOD("set_factor", "factor"), &InputEventMagnifyGesture::set_factor);
	ClassDB::bind_method(D_METHOD("get_factor"), &InputEventMagnifyGesture::get_factor);

	ADD_PROPERTY(PropertyInfo(Variant::REAL, "factor"), "set_factor", "get_factor");
}

InputEventMagnifyGesture::InputEventMagnifyGesture() {

	factor = 1.0;
}
/////////////////////////////

void InputEventPanGesture::set_delta(const Vector2 &p_delta) {

	delta = p_delta;
}

Vector2 InputEventPanGesture::get_delta() const {
	return delta;
}

Ref<InputEvent> InputEventPanGesture::xformed_by(const Transform2D &p_xform, const Vector2 &p_local_ofs) const {

	Ref<InputEventPanGesture> ev;
	ev.instance();

	ev->set_id(get_id());
	ev->set_device(get_device());
	ev->set_modifiers_from_event(this);

	ev->set_position(p_xform.xform(get_position() + p_local_ofs));
	ev->set_delta(get_delta());

	return ev;
}

void InputEventPanGesture::_bind_methods() {

	ClassDB::bind_method(D_METHOD("set_delta", "delta"), &InputEventPanGesture::set_delta);
	ClassDB::bind_method(D_METHOD("get_delta"), &InputEventPanGesture::get_delta);

	ADD_PROPERTY(PropertyInfo(Variant::VECTOR2, "delta"), "set_delta", "get_delta");
}

InputEventPanGesture::InputEventPanGesture() {

	delta = Vector2(0, 0);
}
