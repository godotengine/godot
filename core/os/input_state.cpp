/*************************************************************************/
/*  input.cpp                                                            */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2020 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2020 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "input_state.h"
#include "core/os/os.h"

bool InputState::is_key_pressed(int p_scancode) const {
	return keys_pressed.has(p_scancode);
}

bool InputState::is_mouse_button_pressed(int p_button) const {
	return (mouse_button_mask & (1 << (p_button - 1))) != 0;
}

static int _combine_device(int p_value, int p_device) {
	return p_value | (p_device << 20);
}

bool InputState::is_joy_button_pressed(int p_device, int p_button) const {
	return joy_buttons_pressed.has(_combine_device(p_button, p_device));
}

bool InputState::is_action_pressed(const StringName &p_action) const {
	return action_state.has(p_action) && action_state[p_action].pressed;
}

bool InputState::is_action_just_pressed(const StringName &p_action) const {
	const Map<StringName, InputState::Action>::Element *E = action_state.find(p_action);
	if (!E)
		return false;

	if (Engine::get_singleton()->is_in_physics_frame()) {
		return E->get().pressed && (E->get().exact || E->get().physics_frame == Engine::get_singleton()->get_physics_frames());
	} else {
		return E->get().pressed && (E->get().exact || E->get().idle_frame == Engine::get_singleton()->get_idle_frames());
	}
}

bool InputState::is_action_just_released(const StringName &p_action) const {
	const Map<StringName, InputState::Action>::Element *E = action_state.find(p_action);
	if (!E)
		return false;

	if (Engine::get_singleton()->is_in_physics_frame()) {
		return !E->get().pressed && (E->get().exact || E->get().physics_frame == Engine::get_singleton()->get_physics_frames());
	} else {
		return !E->get().pressed && (E->get().exact || E->get().idle_frame == Engine::get_singleton()->get_idle_frames());
	}
}

float InputState::get_action_strength(const StringName &p_action) const {
	const Map<StringName, InputState::Action>::Element *E = action_state.find(p_action);
	if (!E)
		return 0.0f;
	return E->get().strength;
}

float InputState::get_joy_axis(int p_device, int p_axis) const {
	int c = _combine_device(p_axis, p_device);
	if (joy_axis.has(c)) {
		return joy_axis[c];
	} else {
		return 0;
	}
}

Vector2 InputState::get_joy_vibration_strength(int p_device) {
	if (joy_vibration.has(p_device)) {
		return Vector2(joy_vibration[p_device].weak_magnitude, joy_vibration[p_device].strong_magnitude);
	} else {
		return Vector2(0, 0);
	}
}

uint64_t InputState::get_joy_vibration_timestamp(int p_device) {
	if (joy_vibration.has(p_device)) {
		return joy_vibration[p_device].timestamp;
	} else {
		return 0;
	}
}

float InputState::get_joy_vibration_duration(int p_device) {
	if (joy_vibration.has(p_device)) {
		return joy_vibration[p_device].duration;
	} else {
		return 0.f;
	}
}

Vector3 InputState::get_gravity() const {
	return gravity;
}

Vector3 InputState::get_accelerometer() const {
	return accelerometer;
}

Vector3 InputState::get_magnetometer() const {
	return magnetometer;
}

Vector3 InputState::get_gyroscope() const {
	return gyroscope;
}

void InputState::set_joy_axis(int p_device, int p_axis, float p_value) {
	int c = _combine_device(p_axis, p_device);
	joy_axis[c] = p_value;
}

void InputState::start_joy_vibration(int p_device, float p_weak_magnitude, float p_strong_magnitude, float p_duration) {
	if (p_weak_magnitude < 0.f || p_weak_magnitude > 1.f || p_strong_magnitude < 0.f || p_strong_magnitude > 1.f) {
		return;
	}
	VibrationInfo vibration;
	vibration.weak_magnitude = p_weak_magnitude;
	vibration.strong_magnitude = p_strong_magnitude;
	vibration.duration = p_duration;
	vibration.timestamp = OS::get_singleton()->get_ticks_usec();
	joy_vibration[p_device] = vibration;
}

void InputState::stop_joy_vibration(int p_device) {
	VibrationInfo vibration;
	vibration.weak_magnitude = 0;
	vibration.strong_magnitude = 0;
	vibration.duration = 0;
	vibration.timestamp = OS::get_singleton()->get_ticks_usec();
	joy_vibration[p_device] = vibration;
}

void InputState::set_gravity(const Vector3 &p_gravity) {
	gravity = p_gravity;
}

void InputState::set_accelerometer(const Vector3 &p_accel) {
	accelerometer = p_accel;
}

void InputState::set_magnetometer(const Vector3 &p_magnetometer) {
	magnetometer = p_magnetometer;
}

void InputState::set_gyroscope(const Vector3 &p_gyroscope) {
	gyroscope = p_gyroscope;
}

void InputState::set_mouse_position(const Point2 &p_pos) {
	mouse_pos = p_pos;
}

Point2 InputState::get_mouse_position() const {
	return mouse_pos;
}

// Point2 InputState::get_last_mouse_speed() const {
// 	return mouse_speed_track.speed;
// }

int InputState::get_mouse_button_mask() const {
	return mouse_button_mask; // do not trust OS implementation, should remove it - OS::get_singleton()->get_mouse_button_state();
}

void InputState::action_press(const StringName &p_action, float p_strength) {
	InputState::Action action;
	action.physics_frame = Engine::get_singleton()->get_physics_frames();
	action.idle_frame = Engine::get_singleton()->get_idle_frames();
	action.pressed = true;
	action.strength = p_strength;
	action_state[p_action] = action;
}

void InputState::action_release(const StringName &p_action) {
	InputState::Action action;
	action.physics_frame = Engine::get_singleton()->get_physics_frames();
	action.idle_frame = Engine::get_singleton()->get_idle_frames();
	action.pressed = false;
	action.strength = 0.f;
	action_state[p_action] = action;
}

void InputState::release_pressed_events() {
	keys_pressed.clear();
	joy_buttons_pressed.clear();
	joy_axis.clear();
	mouse_button_mask = 0;

	for (Map<StringName, InputState::Action>::Element *E = action_state.front(); E; E = E->next()) {
		if (E->get().pressed)
			action_release(E->key());
	}
}

void InputState::clear() {
	physics_frame = 0;
	idle_frame = 0;

	keys_pressed.clear();
	joy_buttons_pressed.clear();
	joy_axis.clear();
	joy_vibration.clear();
	mouse_button_mask = 0;

	gravity = Vector3();
	accelerometer = Vector3();
	magnetometer = Vector3();
	gyroscope = Vector3();
	mouse_pos = Vector2();

	action_state.clear();
}

void InputState::feed(const Ref<InputState> &p_state) {
	ERR_FAIL_COND_MSG(p_state.is_null(), "Invalid input state.");

	physics_frame = p_state->physics_frame;
	idle_frame = p_state->idle_frame;

	keys_pressed = p_state->keys_pressed;
	joy_buttons_pressed = p_state->joy_buttons_pressed;
	joy_axis = p_state->joy_axis;
	mouse_button_mask = p_state->mouse_button_mask;
	joy_vibration = p_state->joy_vibration;

	gravity = p_state->gravity;
	accelerometer = p_state->accelerometer;
	magnetometer = p_state->magnetometer;
	gyroscope = p_state->gyroscope;
	mouse_pos = p_state->mouse_pos;

	action_state = p_state->action_state;
}

Dictionary InputState::_get_data() const {

	PoolIntArray r_keys_pressed;
	for (Set<int>::Element *E = keys_pressed.front(); E; E = E->next()) {
		r_keys_pressed.push_back(E->get());
	}
	PoolIntArray r_joy_buttons_pressed;
	for (Set<int>::Element *E = joy_buttons_pressed.front(); E; E = E->next()) {
		r_joy_buttons_pressed.push_back(E->get());
	}
	Dictionary r_action_state;
	for (Map<StringName, InputState::Action>::Element *E = action_state.front(); E; E = E->next()) {
		Dictionary r_action;
		r_action["pressed"] = E->get().pressed;
		// We need a direct way to tell whether an action was pressed or released exactly
		// as both physics and idle frames will be different once de-serialized.
		r_action["exact"] = is_action_just_pressed(E->key()) || is_action_just_released(E->key());
		r_action["strength"] = E->get().strength;
		r_action["physics_frame"] = E->get().physics_frame;
		r_action["idle_frame"] = E->get().idle_frame;
		r_action_state[E->key()] = r_action;
	}
	Dictionary ret;
	ret["physics_frame"] = physics_frame;
	ret["idle_frame"] = idle_frame;

	ret["keys_pressed"] = r_keys_pressed;
	ret["joy_buttons_pressed"] = r_joy_buttons_pressed;
	ret["action_state"] = r_action_state;
	ret["mouse_button_mask"] = mouse_button_mask;

	ret["gravity"] = gravity;
	ret["accelerometer"] = accelerometer;
	ret["magnetometer"] = magnetometer;
	ret["gyroscope"] = gyroscope;
	ret["mouse_pos"] = mouse_pos;

	return ret;
}

void InputState::_set_data(const Dictionary &p_data) {
	if (p_data.has("physics_frame")) {
		physics_frame = p_data["physics_frame"];
	}
	if (p_data.has("idle_frame")) {
		idle_frame = p_data["idle_frame"];
	}
	if (p_data.has("keys_pressed")) {
		keys_pressed.clear();
		PoolIntArray keys = p_data["keys_pressed"];
		for (int i = 0; i < keys.size(); ++i) {
			keys_pressed.insert(keys[i]);
		}
	}
	if (p_data.has("joy_buttons_pressed")) {
		joy_buttons_pressed.clear();
		PoolIntArray buttons = p_data["joy_buttons_pressed"];
		for (int i = 0; i < buttons.size(); ++i) {
			joy_buttons_pressed.insert(buttons[i]);
		}
	}
	if (p_data.has("action_state")) {
		action_state.clear();
		Dictionary action_state_data = p_data["action_state"];
		List<Variant> action_names;
		action_state_data.get_key_list(&action_names);

		for (List<Variant>::Element *E = action_names.front(); E; E = E->next()) {
			InputState::Action action;
			Dictionary action_data = action_state_data[E->get()];

			action.pressed = action_data["pressed"];
			action.exact = action_data["exact"];
			action.strength = action_data["strength"];
			action.physics_frame = action_data["physics_frame"];
			action.idle_frame = action_data["idle_frame"];

			action_state[E->get()] = action;
		}
	}
	if (p_data.has("mouse_button_mask")) {
		gravity = p_data["mouse_button_mask"];
	}
	if (p_data.has("gravity")) {
		gravity = p_data["gravity"];
	}
	if (p_data.has("accelerometer")) {
		accelerometer = p_data["accelerometer"];
	}
	if (p_data.has("magnetometer")) {
		magnetometer = p_data["magnetometer"];
	}
	if (p_data.has("gyroscope")) {
		gyroscope = p_data["gyroscope"];
	}
	if (p_data.has("mouse_pos")) {
		mouse_pos = p_data["mouse_pos"];
	}
}

uint64_t InputState::get_physics_frame() const {
	return physics_frame;
}

uint64_t InputState::get_idle_frame() const {
	return idle_frame;
}

void InputState::_bind_methods() {
	ClassDB::bind_method(D_METHOD("is_key_pressed", "scancode"), &InputState::is_key_pressed);
	ClassDB::bind_method(D_METHOD("is_mouse_button_pressed", "button"), &InputState::is_mouse_button_pressed);
	ClassDB::bind_method(D_METHOD("is_joy_button_pressed", "device", "button"), &InputState::is_joy_button_pressed);
	ClassDB::bind_method(D_METHOD("is_action_pressed", "action"), &InputState::is_action_pressed);
	ClassDB::bind_method(D_METHOD("is_action_just_pressed", "action"), &InputState::is_action_just_pressed);
	ClassDB::bind_method(D_METHOD("is_action_just_released", "action"), &InputState::is_action_just_released);
	ClassDB::bind_method(D_METHOD("get_action_strength", "action"), &InputState::get_action_strength);
	ClassDB::bind_method(D_METHOD("get_joy_axis", "device", "axis"), &InputState::get_joy_axis);
	ClassDB::bind_method(D_METHOD("get_joy_vibration_strength", "device"), &InputState::get_joy_vibration_strength);
	ClassDB::bind_method(D_METHOD("get_joy_vibration_duration", "device"), &InputState::get_joy_vibration_duration);
	ClassDB::bind_method(D_METHOD("start_joy_vibration", "device", "weak_magnitude", "strong_magnitude", "duration"), &InputState::start_joy_vibration, DEFVAL(0));
	ClassDB::bind_method(D_METHOD("stop_joy_vibration", "device"), &InputState::stop_joy_vibration);
	ClassDB::bind_method(D_METHOD("get_gravity"), &InputState::get_gravity);
	ClassDB::bind_method(D_METHOD("get_accelerometer"), &InputState::get_accelerometer);
	ClassDB::bind_method(D_METHOD("get_magnetometer"), &InputState::get_magnetometer);
	ClassDB::bind_method(D_METHOD("get_gyroscope"), &InputState::get_gyroscope);
	// ClassDB::bind_method(D_METHOD("get_last_mouse_speed"), &InputState::get_last_mouse_speed);
	ClassDB::bind_method(D_METHOD("get_mouse_position"), &InputState::get_mouse_position);
	ClassDB::bind_method(D_METHOD("get_mouse_button_mask"), &InputState::get_mouse_button_mask);
	ClassDB::bind_method(D_METHOD("action_press", "action", "strength"), &InputState::action_press, DEFVAL(1.f));
	ClassDB::bind_method(D_METHOD("action_release", "action"), &InputState::action_release);

	ClassDB::bind_method(D_METHOD("feed", "state"), &InputState::feed);
	ClassDB::bind_method(D_METHOD("clear"), &InputState::clear);

	ClassDB::bind_method(D_METHOD("get_physics_frame"), &InputState::get_physics_frame);
	ClassDB::bind_method(D_METHOD("get_idle_frame"), &InputState::get_idle_frame);

	ClassDB::bind_method(D_METHOD("release_pressed_events"), &InputState::release_pressed_events);

	ClassDB::bind_method(D_METHOD("_set_data", "data"), &InputState::_set_data);
	ClassDB::bind_method(D_METHOD("_get_data"), &InputState::_get_data);

	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "gravity"), "", "get_gravity");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "accelerometer"), "", "get_accelerometer");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "magnetometer"), "", "get_magnetometer");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "gyroscope"), "", "get_gyroscope");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR2, "mouse_position"), "", "get_mouse_position");

	ADD_PROPERTY(PropertyInfo(Variant::INT, "physics_frame"), "", "get_physics_frame");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "idle_frame"), "", "get_idle_frame");
	ADD_PROPERTY(PropertyInfo(Variant::DICTIONARY, "data", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_STORAGE), "_set_data", "_get_data");
}

InputState::InputState() {
	physics_frame = 0;
	idle_frame = 0;
	mouse_button_mask = 0;
}
