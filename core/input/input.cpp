/**************************************************************************/
/*  input.cpp                                                             */
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

#include "input.h"

#include "core/config/project_settings.h"
#include "core/input/default_controller_mappings.h"
#include "core/input/input_map.h"
#include "core/os/os.h"

#ifdef DEV_ENABLED
#include "core/os/thread.h"
#endif

static const char *_joy_buttons[(size_t)JoyButton::SDL_MAX] = {
	"a",
	"b",
	"x",
	"y",
	"back",
	"guide",
	"start",
	"leftstick",
	"rightstick",
	"leftshoulder",
	"rightshoulder",
	"dpup",
	"dpdown",
	"dpleft",
	"dpright",
	"misc1",
	"paddle1",
	"paddle2",
	"paddle3",
	"paddle4",
	"touchpad",
};

static const char *_joy_axes[(size_t)JoyAxis::SDL_MAX] = {
	"leftx",
	"lefty",
	"rightx",
	"righty",
	"lefttrigger",
	"righttrigger",
};

Input *Input::singleton = nullptr;

void (*Input::set_mouse_mode_func)(Input::MouseMode) = nullptr;
Input::MouseMode (*Input::get_mouse_mode_func)() = nullptr;
void (*Input::warp_mouse_func)(const Vector2 &p_position) = nullptr;
Input::CursorShape (*Input::get_current_cursor_shape_func)() = nullptr;
void (*Input::set_custom_mouse_cursor_func)(const Ref<Resource> &, Input::CursorShape, const Vector2 &) = nullptr;

Input *Input::get_singleton() {
	return singleton;
}

void Input::set_mouse_mode(MouseMode p_mode) {
	ERR_FAIL_INDEX((int)p_mode, 5);
	set_mouse_mode_func(p_mode);
}

Input::MouseMode Input::get_mouse_mode() const {
	return get_mouse_mode_func();
}

void Input::_bind_methods() {
	ClassDB::bind_method(D_METHOD("is_anything_pressed"), &Input::is_anything_pressed);
	ClassDB::bind_method(D_METHOD("is_key_pressed", "keycode"), &Input::is_key_pressed);
	ClassDB::bind_method(D_METHOD("is_physical_key_pressed", "keycode"), &Input::is_physical_key_pressed);
	ClassDB::bind_method(D_METHOD("is_key_label_pressed", "keycode"), &Input::is_key_label_pressed);
	ClassDB::bind_method(D_METHOD("is_mouse_button_pressed", "button"), &Input::is_mouse_button_pressed);
	ClassDB::bind_method(D_METHOD("is_joy_button_pressed", "device", "button"), &Input::is_joy_button_pressed);
	ClassDB::bind_method(D_METHOD("is_action_pressed", "action", "exact_match"), &Input::is_action_pressed, DEFVAL(false));
	ClassDB::bind_method(D_METHOD("is_action_just_pressed", "action", "exact_match"), &Input::is_action_just_pressed, DEFVAL(false));
	ClassDB::bind_method(D_METHOD("is_action_just_released", "action", "exact_match"), &Input::is_action_just_released, DEFVAL(false));
	ClassDB::bind_method(D_METHOD("get_action_strength", "action", "exact_match"), &Input::get_action_strength, DEFVAL(false));
	ClassDB::bind_method(D_METHOD("get_action_raw_strength", "action", "exact_match"), &Input::get_action_raw_strength, DEFVAL(false));
	ClassDB::bind_method(D_METHOD("get_axis", "negative_action", "positive_action"), &Input::get_axis);
	ClassDB::bind_method(D_METHOD("get_vector", "negative_x", "positive_x", "negative_y", "positive_y", "deadzone"), &Input::get_vector, DEFVAL(-1.0f));
	ClassDB::bind_method(D_METHOD("add_joy_mapping", "mapping", "update_existing"), &Input::add_joy_mapping, DEFVAL(false));
	ClassDB::bind_method(D_METHOD("remove_joy_mapping", "guid"), &Input::remove_joy_mapping);
	ClassDB::bind_method(D_METHOD("is_joy_known", "device"), &Input::is_joy_known);
	ClassDB::bind_method(D_METHOD("get_joy_axis", "device", "axis"), &Input::get_joy_axis);
	ClassDB::bind_method(D_METHOD("get_joy_name", "device"), &Input::get_joy_name);
	ClassDB::bind_method(D_METHOD("get_joy_guid", "device"), &Input::get_joy_guid);
	ClassDB::bind_method(D_METHOD("get_joy_info", "device"), &Input::get_joy_info);
	ClassDB::bind_method(D_METHOD("should_ignore_device", "vendor_id", "product_id"), &Input::should_ignore_device);
	ClassDB::bind_method(D_METHOD("get_connected_joypads"), &Input::get_connected_joypads);
	ClassDB::bind_method(D_METHOD("get_joy_vibration_strength", "device"), &Input::get_joy_vibration_strength);
	ClassDB::bind_method(D_METHOD("get_joy_vibration_duration", "device"), &Input::get_joy_vibration_duration);
	ClassDB::bind_method(D_METHOD("start_joy_vibration", "device", "weak_magnitude", "strong_magnitude", "duration"), &Input::start_joy_vibration, DEFVAL(0));
	ClassDB::bind_method(D_METHOD("stop_joy_vibration", "device"), &Input::stop_joy_vibration);
	ClassDB::bind_method(D_METHOD("vibrate_handheld", "duration_ms"), &Input::vibrate_handheld, DEFVAL(500));
	ClassDB::bind_method(D_METHOD("get_gravity"), &Input::get_gravity);
	ClassDB::bind_method(D_METHOD("get_accelerometer"), &Input::get_accelerometer);
	ClassDB::bind_method(D_METHOD("get_magnetometer"), &Input::get_magnetometer);
	ClassDB::bind_method(D_METHOD("get_gyroscope"), &Input::get_gyroscope);
	ClassDB::bind_method(D_METHOD("set_gravity", "value"), &Input::set_gravity);
	ClassDB::bind_method(D_METHOD("set_accelerometer", "value"), &Input::set_accelerometer);
	ClassDB::bind_method(D_METHOD("set_magnetometer", "value"), &Input::set_magnetometer);
	ClassDB::bind_method(D_METHOD("set_gyroscope", "value"), &Input::set_gyroscope);
	ClassDB::bind_method(D_METHOD("get_last_mouse_velocity"), &Input::get_last_mouse_velocity);
	ClassDB::bind_method(D_METHOD("get_mouse_button_mask"), &Input::get_mouse_button_mask);
	ClassDB::bind_method(D_METHOD("set_mouse_mode", "mode"), &Input::set_mouse_mode);
	ClassDB::bind_method(D_METHOD("get_mouse_mode"), &Input::get_mouse_mode);
	ClassDB::bind_method(D_METHOD("warp_mouse", "position"), &Input::warp_mouse);
	ClassDB::bind_method(D_METHOD("action_press", "action", "strength"), &Input::action_press, DEFVAL(1.f));
	ClassDB::bind_method(D_METHOD("action_release", "action"), &Input::action_release);
	ClassDB::bind_method(D_METHOD("set_default_cursor_shape", "shape"), &Input::set_default_cursor_shape, DEFVAL(CURSOR_ARROW));
	ClassDB::bind_method(D_METHOD("get_current_cursor_shape"), &Input::get_current_cursor_shape);
	ClassDB::bind_method(D_METHOD("set_custom_mouse_cursor", "image", "shape", "hotspot"), &Input::set_custom_mouse_cursor, DEFVAL(CURSOR_ARROW), DEFVAL(Vector2()));
	ClassDB::bind_method(D_METHOD("parse_input_event", "event"), &Input::parse_input_event);
	ClassDB::bind_method(D_METHOD("set_use_accumulated_input", "enable"), &Input::set_use_accumulated_input);
	ClassDB::bind_method(D_METHOD("is_using_accumulated_input"), &Input::is_using_accumulated_input);
	ClassDB::bind_method(D_METHOD("flush_buffered_events"), &Input::flush_buffered_events);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "mouse_mode"), "set_mouse_mode", "get_mouse_mode");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "use_accumulated_input"), "set_use_accumulated_input", "is_using_accumulated_input");

	BIND_ENUM_CONSTANT(MOUSE_MODE_VISIBLE);
	BIND_ENUM_CONSTANT(MOUSE_MODE_HIDDEN);
	BIND_ENUM_CONSTANT(MOUSE_MODE_CAPTURED);
	BIND_ENUM_CONSTANT(MOUSE_MODE_CONFINED);
	BIND_ENUM_CONSTANT(MOUSE_MODE_CONFINED_HIDDEN);

	BIND_ENUM_CONSTANT(CURSOR_ARROW);
	BIND_ENUM_CONSTANT(CURSOR_IBEAM);
	BIND_ENUM_CONSTANT(CURSOR_POINTING_HAND);
	BIND_ENUM_CONSTANT(CURSOR_CROSS);
	BIND_ENUM_CONSTANT(CURSOR_WAIT);
	BIND_ENUM_CONSTANT(CURSOR_BUSY);
	BIND_ENUM_CONSTANT(CURSOR_DRAG);
	BIND_ENUM_CONSTANT(CURSOR_CAN_DROP);
	BIND_ENUM_CONSTANT(CURSOR_FORBIDDEN);
	BIND_ENUM_CONSTANT(CURSOR_VSIZE);
	BIND_ENUM_CONSTANT(CURSOR_HSIZE);
	BIND_ENUM_CONSTANT(CURSOR_BDIAGSIZE);
	BIND_ENUM_CONSTANT(CURSOR_FDIAGSIZE);
	BIND_ENUM_CONSTANT(CURSOR_MOVE);
	BIND_ENUM_CONSTANT(CURSOR_VSPLIT);
	BIND_ENUM_CONSTANT(CURSOR_HSPLIT);
	BIND_ENUM_CONSTANT(CURSOR_HELP);

	ADD_SIGNAL(MethodInfo("joy_connection_changed", PropertyInfo(Variant::INT, "device"), PropertyInfo(Variant::BOOL, "connected")));
}

void Input::get_argument_options(const StringName &p_function, int p_idx, List<String> *r_options) const {
	String pf = p_function;

	if ((p_idx == 0 && (pf == "is_action_pressed" || pf == "action_press" || pf == "action_release" || pf == "is_action_just_pressed" || pf == "is_action_just_released" || pf == "get_action_strength" || pf == "get_action_raw_strength")) ||
			(p_idx < 2 && pf == "get_axis") ||
			(p_idx < 4 && pf == "get_vector")) {
		List<PropertyInfo> pinfo;
		ProjectSettings::get_singleton()->get_property_list(&pinfo);

		for (const PropertyInfo &pi : pinfo) {
			if (!pi.name.begins_with("input/")) {
				continue;
			}

			String name = pi.name.substr(pi.name.find("/") + 1, pi.name.length());
			r_options->push_back(name.quote());
		}
	}
}

void Input::VelocityTrack::update(const Vector2 &p_delta_p) {
	uint64_t tick = OS::get_singleton()->get_ticks_usec();
	uint32_t tdiff = tick - last_tick;
	float delta_t = tdiff / 1000000.0;
	last_tick = tick;

	if (delta_t > max_ref_frame) {
		// First movement in a long time, reset and start again.
		velocity = Vector2();
		accum = p_delta_p;
		accum_t = 0;
		return;
	}

	accum += p_delta_p;
	accum_t += delta_t;

	if (accum_t < min_ref_frame) {
		// Not enough time has passed to calculate speed precisely.
		return;
	}

	velocity = accum / accum_t;
	accum = Vector2();
	accum_t = 0;
}

void Input::VelocityTrack::reset() {
	last_tick = OS::get_singleton()->get_ticks_usec();
	velocity = Vector2();
	accum = Vector2();
	accum_t = 0;
}

Input::VelocityTrack::VelocityTrack() {
	min_ref_frame = 0.1;
	max_ref_frame = 3.0;
	reset();
}

bool Input::is_anything_pressed() const {
	_THREAD_SAFE_METHOD_

	if (!keys_pressed.is_empty() || !joy_buttons_pressed.is_empty() || !mouse_button_mask.is_empty()) {
		return true;
	}

	for (const KeyValue<StringName, Input::Action> &E : action_state) {
		if (E.value.pressed) {
			return true;
		}
	}

	return false;
}

bool Input::is_key_pressed(Key p_keycode) const {
	_THREAD_SAFE_METHOD_
	return keys_pressed.has(p_keycode);
}

bool Input::is_physical_key_pressed(Key p_keycode) const {
	_THREAD_SAFE_METHOD_
	return physical_keys_pressed.has(p_keycode);
}

bool Input::is_key_label_pressed(Key p_keycode) const {
	_THREAD_SAFE_METHOD_
	return key_label_pressed.has(p_keycode);
}

bool Input::is_mouse_button_pressed(MouseButton p_button) const {
	_THREAD_SAFE_METHOD_
	return mouse_button_mask.has_flag(mouse_button_to_mask(p_button));
}

static JoyAxis _combine_device(JoyAxis p_value, int p_device) {
	return JoyAxis((int)p_value | (p_device << 20));
}

static JoyButton _combine_device(JoyButton p_value, int p_device) {
	return JoyButton((int)p_value | (p_device << 20));
}

bool Input::is_joy_button_pressed(int p_device, JoyButton p_button) const {
	_THREAD_SAFE_METHOD_
	return joy_buttons_pressed.has(_combine_device(p_button, p_device));
}

bool Input::is_action_pressed(const StringName &p_action, bool p_exact) const {
	ERR_FAIL_COND_V_MSG(!InputMap::get_singleton()->has_action(p_action), false, InputMap::get_singleton()->suggest_actions(p_action));
	return action_state.has(p_action) && action_state[p_action].pressed > 0 && (p_exact ? action_state[p_action].exact : true);
}

bool Input::is_action_just_pressed(const StringName &p_action, bool p_exact) const {
	ERR_FAIL_COND_V_MSG(!InputMap::get_singleton()->has_action(p_action), false, InputMap::get_singleton()->suggest_actions(p_action));
	HashMap<StringName, Action>::ConstIterator E = action_state.find(p_action);
	if (!E) {
		return false;
	}

	if (p_exact && E->value.exact == false) {
		return false;
	}

	// Backward compatibility for legacy behavior, only return true if currently pressed.
	bool pressed_requirement = legacy_just_pressed_behavior ? E->value.pressed : true;

	if (Engine::get_singleton()->is_in_physics_frame()) {
		return pressed_requirement && E->value.pressed_physics_frame == Engine::get_singleton()->get_physics_frames();
	} else {
		return pressed_requirement && E->value.pressed_process_frame == Engine::get_singleton()->get_process_frames();
	}
}

bool Input::is_action_just_released(const StringName &p_action, bool p_exact) const {
	ERR_FAIL_COND_V_MSG(!InputMap::get_singleton()->has_action(p_action), false, InputMap::get_singleton()->suggest_actions(p_action));
	HashMap<StringName, Action>::ConstIterator E = action_state.find(p_action);
	if (!E) {
		return false;
	}

	if (p_exact && E->value.exact == false) {
		return false;
	}

	// Backward compatibility for legacy behavior, only return true if currently released.
	bool released_requirement = legacy_just_pressed_behavior ? !E->value.pressed : true;

	if (Engine::get_singleton()->is_in_physics_frame()) {
		return released_requirement && E->value.released_physics_frame == Engine::get_singleton()->get_physics_frames();
	} else {
		return released_requirement && E->value.released_process_frame == Engine::get_singleton()->get_process_frames();
	}
}

float Input::get_action_strength(const StringName &p_action, bool p_exact) const {
	ERR_FAIL_COND_V_MSG(!InputMap::get_singleton()->has_action(p_action), 0.0, InputMap::get_singleton()->suggest_actions(p_action));
	HashMap<StringName, Action>::ConstIterator E = action_state.find(p_action);
	if (!E) {
		return 0.0f;
	}

	if (p_exact && E->value.exact == false) {
		return 0.0f;
	}

	return E->value.strength;
}

float Input::get_action_raw_strength(const StringName &p_action, bool p_exact) const {
	ERR_FAIL_COND_V_MSG(!InputMap::get_singleton()->has_action(p_action), 0.0, InputMap::get_singleton()->suggest_actions(p_action));
	HashMap<StringName, Action>::ConstIterator E = action_state.find(p_action);
	if (!E) {
		return 0.0f;
	}

	if (p_exact && E->value.exact == false) {
		return 0.0f;
	}

	return E->value.raw_strength;
}

float Input::get_axis(const StringName &p_negative_action, const StringName &p_positive_action) const {
	return get_action_strength(p_positive_action) - get_action_strength(p_negative_action);
}

Vector2 Input::get_vector(const StringName &p_negative_x, const StringName &p_positive_x, const StringName &p_negative_y, const StringName &p_positive_y, float p_deadzone) const {
	Vector2 vector = Vector2(
			get_action_raw_strength(p_positive_x) - get_action_raw_strength(p_negative_x),
			get_action_raw_strength(p_positive_y) - get_action_raw_strength(p_negative_y));

	if (p_deadzone < 0.0f) {
		// If the deadzone isn't specified, get it from the average of the actions.
		p_deadzone = 0.25 *
				(InputMap::get_singleton()->action_get_deadzone(p_positive_x) +
						InputMap::get_singleton()->action_get_deadzone(p_negative_x) +
						InputMap::get_singleton()->action_get_deadzone(p_positive_y) +
						InputMap::get_singleton()->action_get_deadzone(p_negative_y));
	}

	// Circular length limiting and deadzone.
	float length = vector.length();
	if (length <= p_deadzone) {
		return Vector2();
	} else if (length > 1.0f) {
		return vector / length;
	} else {
		// Inverse lerp length to map (p_deadzone, 1) to (0, 1).
		return vector * (Math::inverse_lerp(p_deadzone, 1.0f, length) / length);
	}
}

float Input::get_joy_axis(int p_device, JoyAxis p_axis) const {
	_THREAD_SAFE_METHOD_
	JoyAxis c = _combine_device(p_axis, p_device);
	if (_joy_axis.has(c)) {
		return _joy_axis[c];
	} else {
		return 0;
	}
}

String Input::get_joy_name(int p_idx) {
	_THREAD_SAFE_METHOD_
	return joy_names[p_idx].name;
}

Vector2 Input::get_joy_vibration_strength(int p_device) {
	if (joy_vibration.has(p_device)) {
		return Vector2(joy_vibration[p_device].weak_magnitude, joy_vibration[p_device].strong_magnitude);
	} else {
		return Vector2(0, 0);
	}
}

uint64_t Input::get_joy_vibration_timestamp(int p_device) {
	if (joy_vibration.has(p_device)) {
		return joy_vibration[p_device].timestamp;
	} else {
		return 0;
	}
}

float Input::get_joy_vibration_duration(int p_device) {
	if (joy_vibration.has(p_device)) {
		return joy_vibration[p_device].duration;
	} else {
		return 0.f;
	}
}

static String _hex_str(uint8_t p_byte) {
	static const char *dict = "0123456789abcdef";
	char ret[3];
	ret[2] = 0;

	ret[0] = dict[p_byte >> 4];
	ret[1] = dict[p_byte & 0xf];

	return ret;
}

void Input::joy_connection_changed(int p_idx, bool p_connected, String p_name, String p_guid, Dictionary p_joypad_info) {
	_THREAD_SAFE_METHOD_
	Joypad js;
	js.name = p_connected ? p_name : "";
	js.uid = p_connected ? p_guid : "";
	js.info = p_connected ? p_joypad_info : Dictionary();

	if (p_connected) {
		String uidname = p_guid;
		if (p_guid.is_empty()) {
			int uidlen = MIN(p_name.length(), 16);
			for (int i = 0; i < uidlen; i++) {
				uidname = uidname + _hex_str(p_name[i]);
			}
		}
		js.uid = uidname;
		js.connected = true;
		int mapping = fallback_mapping;
		for (int i = 0; i < map_db.size(); i++) {
			if (js.uid == map_db[i].uid) {
				mapping = i;
				js.name = map_db[i].name;
			}
		}
		js.mapping = mapping;
	} else {
		js.connected = false;
		for (int i = 0; i < (int)JoyButton::MAX; i++) {
			JoyButton c = _combine_device((JoyButton)i, p_idx);
			joy_buttons_pressed.erase(c);
		}
		for (int i = 0; i < (int)JoyAxis::MAX; i++) {
			set_joy_axis(p_idx, (JoyAxis)i, 0.0f);
		}
	}
	joy_names[p_idx] = js;

	// Ensure this signal is emitted on the main thread, as some platforms (e.g. Linux) call this from a different thread.
	call_deferred("emit_signal", SNAME("joy_connection_changed"), p_idx, p_connected);
}

Vector3 Input::get_gravity() const {
	_THREAD_SAFE_METHOD_
	return gravity;
}

Vector3 Input::get_accelerometer() const {
	_THREAD_SAFE_METHOD_
	return accelerometer;
}

Vector3 Input::get_magnetometer() const {
	_THREAD_SAFE_METHOD_
	return magnetometer;
}

Vector3 Input::get_gyroscope() const {
	_THREAD_SAFE_METHOD_
	return gyroscope;
}

void Input::_parse_input_event_impl(const Ref<InputEvent> &p_event, bool p_is_emulated) {
	// This function does the final delivery of the input event to user land.
	// Regardless where the event came from originally, this has to happen on the main thread.
	DEV_ASSERT(Thread::get_caller_id() == Thread::get_main_id());

	// Notes on mouse-touch emulation:
	// - Emulated mouse events are parsed, that is, re-routed to this method, so they make the same effects
	//   as true mouse events. The only difference is the situation is flagged as emulated so they are not
	//   emulated back to touch events in an endless loop.
	// - Emulated touch events are handed right to the main loop (i.e., the SceneTree) because they don't
	//   require additional handling by this class.

	Ref<InputEventKey> k = p_event;
	if (k.is_valid() && !k->is_echo() && k->get_keycode() != Key::NONE) {
		if (k->is_pressed()) {
			keys_pressed.insert(k->get_keycode());
		} else {
			keys_pressed.erase(k->get_keycode());
		}
	}
	if (k.is_valid() && !k->is_echo() && k->get_physical_keycode() != Key::NONE) {
		if (k->is_pressed()) {
			physical_keys_pressed.insert(k->get_physical_keycode());
		} else {
			physical_keys_pressed.erase(k->get_physical_keycode());
		}
	}
	if (k.is_valid() && !k->is_echo() && k->get_key_label() != Key::NONE) {
		if (k->is_pressed()) {
			key_label_pressed.insert(k->get_key_label());
		} else {
			key_label_pressed.erase(k->get_key_label());
		}
	}

	Ref<InputEventMouseButton> mb = p_event;

	if (mb.is_valid()) {
		if (mb->is_pressed()) {
			mouse_button_mask.set_flag(mouse_button_to_mask(mb->get_button_index()));
		} else {
			mouse_button_mask.clear_flag(mouse_button_to_mask(mb->get_button_index()));
		}

		Point2 pos = mb->get_global_position();
		if (mouse_pos != pos) {
			set_mouse_position(pos);
		}

		if (event_dispatch_function && emulate_touch_from_mouse && !p_is_emulated && mb->get_button_index() == MouseButton::LEFT) {
			Ref<InputEventScreenTouch> touch_event;
			touch_event.instantiate();
			touch_event->set_pressed(mb->is_pressed());
			touch_event->set_canceled(mb->is_canceled());
			touch_event->set_position(mb->get_position());
			touch_event->set_double_tap(mb->is_double_click());
			touch_event->set_device(InputEvent::DEVICE_ID_EMULATION);
			_THREAD_SAFE_UNLOCK_
			event_dispatch_function(touch_event);
			_THREAD_SAFE_LOCK_
		}
	}

	Ref<InputEventMouseMotion> mm = p_event;

	if (mm.is_valid()) {
		Point2 position = mm->get_global_position();
		if (mouse_pos != position) {
			set_mouse_position(position);
		}
		Vector2 relative = mm->get_relative();
		mouse_velocity_track.update(relative);

		if (event_dispatch_function && emulate_touch_from_mouse && !p_is_emulated && mm->get_button_mask().has_flag(MouseButtonMask::LEFT)) {
			Ref<InputEventScreenDrag> drag_event;
			drag_event.instantiate();

			drag_event->set_position(position);
			drag_event->set_relative(relative);
			drag_event->set_tilt(mm->get_tilt());
			drag_event->set_pen_inverted(mm->get_pen_inverted());
			drag_event->set_pressure(mm->get_pressure());
			drag_event->set_velocity(get_last_mouse_velocity());
			drag_event->set_device(InputEvent::DEVICE_ID_EMULATION);

			_THREAD_SAFE_UNLOCK_
			event_dispatch_function(drag_event);
			_THREAD_SAFE_LOCK_
		}
	}

	Ref<InputEventScreenTouch> st = p_event;

	if (st.is_valid()) {
		if (st->is_pressed()) {
			VelocityTrack &track = touch_velocity_track[st->get_index()];
			track.reset();
		} else {
			// Since a pointer index may not occur again (OSs may or may not reuse them),
			// imperatively remove it from the map to keep no fossil entries in it
			touch_velocity_track.erase(st->get_index());
		}

		if (emulate_mouse_from_touch) {
			bool translate = false;
			if (st->is_pressed()) {
				if (mouse_from_touch_index == -1) {
					translate = true;
					mouse_from_touch_index = st->get_index();
				}
			} else {
				if (st->get_index() == mouse_from_touch_index) {
					translate = true;
					mouse_from_touch_index = -1;
				}
			}

			if (translate) {
				Ref<InputEventMouseButton> button_event;
				button_event.instantiate();

				button_event->set_device(InputEvent::DEVICE_ID_EMULATION);
				button_event->set_position(st->get_position());
				button_event->set_global_position(st->get_position());
				button_event->set_pressed(st->is_pressed());
				button_event->set_canceled(st->is_canceled());
				button_event->set_button_index(MouseButton::LEFT);
				button_event->set_double_click(st->is_double_tap());

				BitField<MouseButtonMask> ev_bm = mouse_button_mask;
				if (st->is_pressed()) {
					ev_bm.set_flag(MouseButtonMask::LEFT);
				} else {
					ev_bm.clear_flag(MouseButtonMask::LEFT);
				}
				button_event->set_button_mask(ev_bm);

				_parse_input_event_impl(button_event, true);
			}
		}
	}

	Ref<InputEventScreenDrag> sd = p_event;

	if (sd.is_valid()) {
		VelocityTrack &track = touch_velocity_track[sd->get_index()];
		track.update(sd->get_relative());
		sd->set_velocity(track.velocity);

		if (emulate_mouse_from_touch && sd->get_index() == mouse_from_touch_index) {
			Ref<InputEventMouseMotion> motion_event;
			motion_event.instantiate();

			motion_event->set_device(InputEvent::DEVICE_ID_EMULATION);
			motion_event->set_tilt(sd->get_tilt());
			motion_event->set_pen_inverted(sd->get_pen_inverted());
			motion_event->set_pressure(sd->get_pressure());
			motion_event->set_position(sd->get_position());
			motion_event->set_global_position(sd->get_position());
			motion_event->set_relative(sd->get_relative());
			motion_event->set_velocity(sd->get_velocity());
			motion_event->set_button_mask(mouse_button_mask);

			_parse_input_event_impl(motion_event, true);
		}
	}

	Ref<InputEventJoypadButton> jb = p_event;

	if (jb.is_valid()) {
		JoyButton c = _combine_device(jb->get_button_index(), jb->get_device());

		if (jb->is_pressed()) {
			joy_buttons_pressed.insert(c);
		} else {
			joy_buttons_pressed.erase(c);
		}
	}

	Ref<InputEventJoypadMotion> jm = p_event;

	if (jm.is_valid()) {
		set_joy_axis(jm->get_device(), jm->get_axis(), jm->get_axis_value());
	}

	Ref<InputEventGesture> ge = p_event;

	if (ge.is_valid()) {
		if (event_dispatch_function) {
			_THREAD_SAFE_UNLOCK_
			event_dispatch_function(ge);
			_THREAD_SAFE_LOCK_
		}
	}

	for (const KeyValue<StringName, InputMap::Action> &E : InputMap::get_singleton()->get_action_map()) {
		const int event_index = InputMap::get_singleton()->event_get_index(p_event, E.key);
		if (event_index == -1) {
			continue;
		}

		Action &action = action_state[E.key];
		if (!p_event->is_echo()) {
			if (p_event->is_action_pressed(E.key)) {
				if (!action.pressed) {
					action.pressed_physics_frame = Engine::get_singleton()->get_physics_frames();
					action.pressed_process_frame = Engine::get_singleton()->get_process_frames();
				}
				action.pressed |= ((uint64_t)1 << event_index);
			} else {
				action.pressed &= ~((uint64_t)1 << event_index);
				action.pressed &= ~(1 << MAX_EVENT); // Always release the event from action_press() method.

				if (!action.pressed) {
					action.released_physics_frame = Engine::get_singleton()->get_physics_frames();
					action.released_process_frame = Engine::get_singleton()->get_process_frames();
				}
				_update_action_strength(action, MAX_EVENT, 0.0);
				_update_action_raw_strength(action, MAX_EVENT, 0.0);
			}
			action.exact = InputMap::get_singleton()->event_is_action(p_event, E.key, true);
		}
		_update_action_strength(action, event_index, p_event->get_action_strength(E.key));
		_update_action_raw_strength(action, event_index, p_event->get_action_raw_strength(E.key));
	}

	if (event_dispatch_function) {
		_THREAD_SAFE_UNLOCK_
		event_dispatch_function(p_event);
		_THREAD_SAFE_LOCK_
	}
}

void Input::set_joy_axis(int p_device, JoyAxis p_axis, float p_value) {
	_THREAD_SAFE_METHOD_
	JoyAxis c = _combine_device(p_axis, p_device);
	_joy_axis[c] = p_value;
}

void Input::start_joy_vibration(int p_device, float p_weak_magnitude, float p_strong_magnitude, float p_duration) {
	_THREAD_SAFE_METHOD_
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

void Input::stop_joy_vibration(int p_device) {
	_THREAD_SAFE_METHOD_
	VibrationInfo vibration;
	vibration.weak_magnitude = 0;
	vibration.strong_magnitude = 0;
	vibration.duration = 0;
	vibration.timestamp = OS::get_singleton()->get_ticks_usec();
	joy_vibration[p_device] = vibration;
}

void Input::vibrate_handheld(int p_duration_ms) {
	OS::get_singleton()->vibrate_handheld(p_duration_ms);
}

void Input::set_gravity(const Vector3 &p_gravity) {
	_THREAD_SAFE_METHOD_

	gravity = p_gravity;
}

void Input::set_accelerometer(const Vector3 &p_accel) {
	_THREAD_SAFE_METHOD_

	accelerometer = p_accel;
}

void Input::set_magnetometer(const Vector3 &p_magnetometer) {
	_THREAD_SAFE_METHOD_

	magnetometer = p_magnetometer;
}

void Input::set_gyroscope(const Vector3 &p_gyroscope) {
	_THREAD_SAFE_METHOD_

	gyroscope = p_gyroscope;
}

void Input::set_mouse_position(const Point2 &p_posf) {
	mouse_pos = p_posf;
}

Point2 Input::get_mouse_position() const {
	return mouse_pos;
}

Point2 Input::get_last_mouse_velocity() {
	mouse_velocity_track.update(Vector2());
	return mouse_velocity_track.velocity;
}

BitField<MouseButtonMask> Input::get_mouse_button_mask() const {
	return mouse_button_mask; // do not trust OS implementation, should remove it - OS::get_singleton()->get_mouse_button_state();
}

void Input::warp_mouse(const Vector2 &p_position) {
	warp_mouse_func(p_position);
}

Point2i Input::warp_mouse_motion(const Ref<InputEventMouseMotion> &p_motion, const Rect2 &p_rect) {
	// The relative distance reported for the next event after a warp is in the boundaries of the
	// size of the rect on that axis, but it may be greater, in which case there's no problem as fmod()
	// will warp it, but if the pointer has moved in the opposite direction between the pointer relocation
	// and the subsequent event, the reported relative distance will be less than the size of the rect
	// and thus fmod() will be disabled for handling the situation.
	// And due to this mouse warping mechanism being stateless, we need to apply some heuristics to
	// detect the warp: if the relative distance is greater than the half of the size of the relevant rect
	// (checked per each axis), it will be considered as the consequence of a former pointer warp.

	const Point2i rel_sign(p_motion->get_relative().x >= 0.0f ? 1 : -1, p_motion->get_relative().y >= 0.0 ? 1 : -1);
	const Size2i warp_margin = p_rect.size * 0.5f;
	const Point2i rel_warped(
			Math::fmod(p_motion->get_relative().x + rel_sign.x * warp_margin.x, p_rect.size.x) - rel_sign.x * warp_margin.x,
			Math::fmod(p_motion->get_relative().y + rel_sign.y * warp_margin.y, p_rect.size.y) - rel_sign.y * warp_margin.y);

	const Point2i pos_local = p_motion->get_global_position() - p_rect.position;
	const Point2i pos_warped(Math::fposmod(pos_local.x, p_rect.size.x), Math::fposmod(pos_local.y, p_rect.size.y));
	if (pos_warped != pos_local) {
		warp_mouse(pos_warped + p_rect.position);
	}

	return rel_warped;
}

void Input::action_press(const StringName &p_action, float p_strength) {
	// Create or retrieve existing action.
	Action &action = action_state[p_action];

	if (!action.pressed) {
		action.pressed_physics_frame = Engine::get_singleton()->get_physics_frames();
		action.pressed_process_frame = Engine::get_singleton()->get_process_frames();
	}
	action.pressed |= 1 << MAX_EVENT;
	_update_action_strength(action, MAX_EVENT, p_strength);
	_update_action_raw_strength(action, MAX_EVENT, p_strength);
	action.exact = true;
}

void Input::action_release(const StringName &p_action) {
	// Create or retrieve existing action.
	Action &action = action_state[p_action];

	action.pressed = 0;
	action.strength = 0.0;
	action.raw_strength = 0.0;
	action.released_physics_frame = Engine::get_singleton()->get_physics_frames();
	action.released_process_frame = Engine::get_singleton()->get_process_frames();
	for (uint64_t i = 0; i <= MAX_EVENT; i++) {
		action.strengths[i] = 0.0;
		action.raw_strengths[i] = 0.0;
	}
	action.exact = true;
}

void Input::set_emulate_touch_from_mouse(bool p_emulate) {
	emulate_touch_from_mouse = p_emulate;
}

bool Input::is_emulating_touch_from_mouse() const {
	return emulate_touch_from_mouse;
}

// Calling this whenever the game window is focused helps unsticking the "touch mouse"
// if the OS or its abstraction class hasn't properly reported that touch pointers raised
void Input::ensure_touch_mouse_raised() {
	_THREAD_SAFE_METHOD_
	if (mouse_from_touch_index != -1) {
		mouse_from_touch_index = -1;

		Ref<InputEventMouseButton> button_event;
		button_event.instantiate();

		button_event->set_device(InputEvent::DEVICE_ID_EMULATION);
		button_event->set_position(mouse_pos);
		button_event->set_global_position(mouse_pos);
		button_event->set_pressed(false);
		button_event->set_button_index(MouseButton::LEFT);
		BitField<MouseButtonMask> ev_bm = mouse_button_mask;
		ev_bm.clear_flag(MouseButtonMask::LEFT);
		button_event->set_button_mask(ev_bm);

		_parse_input_event_impl(button_event, true);
	}
}

void Input::set_emulate_mouse_from_touch(bool p_emulate) {
	emulate_mouse_from_touch = p_emulate;
}

bool Input::is_emulating_mouse_from_touch() const {
	return emulate_mouse_from_touch;
}

Input::CursorShape Input::get_default_cursor_shape() const {
	return default_shape;
}

void Input::set_default_cursor_shape(CursorShape p_shape) {
	if (default_shape == p_shape) {
		return;
	}

	default_shape = p_shape;
	// The default shape is set in Viewport::_gui_input_event. To instantly
	// see the shape in the viewport we need to trigger a mouse motion event.
	Ref<InputEventMouseMotion> mm;
	mm.instantiate();
	mm->set_position(mouse_pos);
	mm->set_global_position(mouse_pos);
	mm->set_device(InputEvent::DEVICE_ID_INTERNAL);
	parse_input_event(mm);
}

Input::CursorShape Input::get_current_cursor_shape() const {
	return get_current_cursor_shape_func();
}

void Input::set_custom_mouse_cursor(const Ref<Resource> &p_cursor, CursorShape p_shape, const Vector2 &p_hotspot) {
	if (Engine::get_singleton()->is_editor_hint()) {
		return;
	}

	ERR_FAIL_INDEX(p_shape, CursorShape::CURSOR_MAX);

	set_custom_mouse_cursor_func(p_cursor, p_shape, p_hotspot);
}

void Input::parse_input_event(const Ref<InputEvent> &p_event) {
	_THREAD_SAFE_METHOD_

	ERR_FAIL_COND(p_event.is_null());

#ifdef DEBUG_ENABLED
	uint64_t curr_frame = Engine::get_singleton()->get_process_frames();
	if (curr_frame != last_parsed_frame) {
		frame_parsed_events.clear();
		last_parsed_frame = curr_frame;
		frame_parsed_events.insert(p_event);
	} else if (frame_parsed_events.has(p_event)) {
		// It would be technically safe to send the same event in cases such as:
		// - After an explicit flush.
		// - In platforms using buffering when agile flushing is enabled, after one of the mid-frame flushes.
		// - If platform doesn't use buffering and event accumulation is disabled.
		// - If platform doesn't use buffering and the event type is not accumulable.
		// However, it wouldn't be reasonable to ask users to remember the full ruleset and be aware at all times
		// of the possibilities of the target platform, project settings and engine internals, which may change
		// without prior notice.
		// Therefore, the guideline is, "don't send the same event object more than once per frame".
		WARN_PRINT_ONCE(
				"An input event object is being parsed more than once in the same frame, which is unsafe.\n"
				"If you are generating events in a script, you have to instantiate a new event instead of sending the same one more than once, unless the original one was sent on an earlier frame.\n"
				"You can call duplicate() on the event to get a new instance with identical values.");
	} else {
		frame_parsed_events.insert(p_event);
	}
#endif

	if (use_accumulated_input) {
		if (buffered_events.is_empty() || !buffered_events.back()->get()->accumulate(p_event)) {
			buffered_events.push_back(p_event);
		}
	} else if (use_input_buffering) {
		buffered_events.push_back(p_event);
	} else {
		_parse_input_event_impl(p_event, false);
	}
}

void Input::flush_buffered_events() {
	_THREAD_SAFE_METHOD_

	while (buffered_events.front()) {
		// The final delivery of the input event involves releasing the lock.
		// While the lock is released, another thread may lock it and add new events to the back.
		// Therefore, we get each event and pop it while we still have the lock,
		// to ensure the list is in a consistent state.
		List<Ref<InputEvent>>::Element *E = buffered_events.front();
		Ref<InputEvent> e = E->get();
		buffered_events.pop_front();

		_parse_input_event_impl(e, false);
	}
}

bool Input::is_using_input_buffering() {
	return use_input_buffering;
}

void Input::set_use_input_buffering(bool p_enable) {
	use_input_buffering = p_enable;
}

void Input::set_use_accumulated_input(bool p_enable) {
	use_accumulated_input = p_enable;
}

bool Input::is_using_accumulated_input() {
	return use_accumulated_input;
}

void Input::release_pressed_events() {
	flush_buffered_events(); // this is needed to release actions strengths

	keys_pressed.clear();
	physical_keys_pressed.clear();
	key_label_pressed.clear();
	joy_buttons_pressed.clear();
	_joy_axis.clear();

	for (KeyValue<StringName, Input::Action> &E : action_state) {
		if (E.value.pressed > 0) {
			// Make sure the action is really released.
			E.value.pressed = 1;
			action_release(E.key);
		}
	}
}

void Input::set_event_dispatch_function(EventDispatchFunc p_function) {
	event_dispatch_function = p_function;
}

void Input::joy_button(int p_device, JoyButton p_button, bool p_pressed) {
	_THREAD_SAFE_METHOD_;
	Joypad &joy = joy_names[p_device];
	ERR_FAIL_INDEX((int)p_button, (int)JoyButton::MAX);

	if (joy.last_buttons[(size_t)p_button] == p_pressed) {
		return;
	}
	joy.last_buttons[(size_t)p_button] = p_pressed;
	if (joy.mapping == -1) {
		_button_event(p_device, p_button, p_pressed);
		return;
	}

	JoyEvent map = _get_mapped_button_event(map_db[joy.mapping], p_button);

	if (map.type == TYPE_BUTTON) {
		_button_event(p_device, (JoyButton)map.index, p_pressed);
		return;
	}

	if (map.type == TYPE_AXIS) {
		_axis_event(p_device, (JoyAxis)map.index, p_pressed ? map.value : 0.0);
	}
	// no event?
}

void Input::joy_axis(int p_device, JoyAxis p_axis, float p_value) {
	_THREAD_SAFE_METHOD_;

	ERR_FAIL_INDEX((int)p_axis, (int)JoyAxis::MAX);

	Joypad &joy = joy_names[p_device];

	if (joy.last_axis[(size_t)p_axis] == p_value) {
		return;
	}

	joy.last_axis[(size_t)p_axis] = p_value;

	if (joy.mapping == -1) {
		_axis_event(p_device, p_axis, p_value);
		return;
	}

	JoyAxisRange range;
	JoyEvent map = _get_mapped_axis_event(map_db[joy.mapping], p_axis, p_value, range);

	if (map.type == TYPE_BUTTON) {
		bool pressed = map.value > 0.5;
		if (pressed != joy_buttons_pressed.has(_combine_device((JoyButton)map.index, p_device))) {
			_button_event(p_device, (JoyButton)map.index, pressed);
		}

		// Ensure opposite D-Pad button is also released.
		switch ((JoyButton)map.index) {
			case JoyButton::DPAD_UP:
				if (joy_buttons_pressed.has(_combine_device(JoyButton::DPAD_DOWN, p_device))) {
					_button_event(p_device, JoyButton::DPAD_DOWN, false);
				}
				break;
			case JoyButton::DPAD_DOWN:
				if (joy_buttons_pressed.has(_combine_device(JoyButton::DPAD_UP, p_device))) {
					_button_event(p_device, JoyButton::DPAD_UP, false);
				}
				break;
			case JoyButton::DPAD_LEFT:
				if (joy_buttons_pressed.has(_combine_device(JoyButton::DPAD_RIGHT, p_device))) {
					_button_event(p_device, JoyButton::DPAD_RIGHT, false);
				}
				break;
			case JoyButton::DPAD_RIGHT:
				if (joy_buttons_pressed.has(_combine_device(JoyButton::DPAD_LEFT, p_device))) {
					_button_event(p_device, JoyButton::DPAD_LEFT, false);
				}
				break;
			default:
				// Nothing to do.
				break;
		}
		return;
	}

	if (map.type == TYPE_AXIS) {
		JoyAxis axis = JoyAxis(map.index);
		float value = map.value;
		if (range == FULL_AXIS && (axis == JoyAxis::TRIGGER_LEFT || axis == JoyAxis::TRIGGER_RIGHT)) {
			// Convert to a value between 0.0f and 1.0f.
			value = 0.5f + value / 2.0f;
		}
		_axis_event(p_device, axis, value);
		return;
	}
}

void Input::joy_hat(int p_device, BitField<HatMask> p_val) {
	_THREAD_SAFE_METHOD_;
	const Joypad &joy = joy_names[p_device];

	JoyEvent map[(size_t)HatDir::MAX];

	map[(size_t)HatDir::UP].type = TYPE_BUTTON;
	map[(size_t)HatDir::UP].index = (int)JoyButton::DPAD_UP;
	map[(size_t)HatDir::UP].value = 0;

	map[(size_t)HatDir::RIGHT].type = TYPE_BUTTON;
	map[(size_t)HatDir::RIGHT].index = (int)JoyButton::DPAD_RIGHT;
	map[(size_t)HatDir::RIGHT].value = 0;

	map[(size_t)HatDir::DOWN].type = TYPE_BUTTON;
	map[(size_t)HatDir::DOWN].index = (int)JoyButton::DPAD_DOWN;
	map[(size_t)HatDir::DOWN].value = 0;

	map[(size_t)HatDir::LEFT].type = TYPE_BUTTON;
	map[(size_t)HatDir::LEFT].index = (int)JoyButton::DPAD_LEFT;
	map[(size_t)HatDir::LEFT].value = 0;

	if (joy.mapping != -1) {
		_get_mapped_hat_events(map_db[joy.mapping], (HatDir)0, map);
	}

	int cur_val = joy_names[p_device].hat_current;

	for (int hat_direction = 0, hat_mask = 1; hat_direction < (int)HatDir::MAX; hat_direction++, hat_mask <<= 1) {
		if (((int)p_val & hat_mask) != (cur_val & hat_mask)) {
			if (map[hat_direction].type == TYPE_BUTTON) {
				_button_event(p_device, (JoyButton)map[hat_direction].index, (int)p_val & hat_mask);
			}
			if (map[hat_direction].type == TYPE_AXIS) {
				_axis_event(p_device, (JoyAxis)map[hat_direction].index, ((int)p_val & hat_mask) ? map[hat_direction].value : 0.0);
			}
		}
	}

	joy_names[p_device].hat_current = (int)p_val;
}

void Input::_button_event(int p_device, JoyButton p_index, bool p_pressed) {
	Ref<InputEventJoypadButton> ievent;
	ievent.instantiate();
	ievent->set_device(p_device);
	ievent->set_button_index(p_index);
	ievent->set_pressed(p_pressed);

	parse_input_event(ievent);
}

void Input::_axis_event(int p_device, JoyAxis p_axis, float p_value) {
	Ref<InputEventJoypadMotion> ievent;
	ievent.instantiate();
	ievent->set_device(p_device);
	ievent->set_axis(p_axis);
	ievent->set_axis_value(p_value);

	parse_input_event(ievent);
}

void Input::_update_action_strength(Action &p_action, int p_event_index, float p_strength) {
	ERR_FAIL_INDEX(p_event_index, (int)MAX_EVENT + 1);

	float old_strength = p_action.strengths[p_event_index];
	p_action.strengths[p_event_index] = p_strength;

	if (p_strength > p_action.strength) {
		p_action.strength = p_strength;
	} else if (Math::is_equal_approx(old_strength, p_action.strength)) {
		p_action.strength = p_strength;
		for (uint64_t i = 0; i <= MAX_EVENT; i++) {
			p_action.strength = MAX(p_action.strength, p_action.strengths[i]);
		}
	}
}

void Input::_update_action_raw_strength(Action &p_action, int p_event_index, float p_strength) {
	ERR_FAIL_INDEX(p_event_index, (int)MAX_EVENT + 1);

	float old_strength = p_action.raw_strengths[p_event_index];
	p_action.raw_strengths[p_event_index] = p_strength;

	if (p_strength > p_action.raw_strength) {
		p_action.raw_strength = p_strength;
	} else if (Math::is_equal_approx(old_strength, p_action.raw_strength)) {
		p_action.raw_strength = p_strength;
		for (uint64_t i = 0; i <= MAX_EVENT; i++) {
			p_action.raw_strength = MAX(p_action.raw_strength, p_action.raw_strengths[i]);
		}
	}
}

Input::JoyEvent Input::_get_mapped_button_event(const JoyDeviceMapping &mapping, JoyButton p_button) {
	JoyEvent event;

	for (int i = 0; i < mapping.bindings.size(); i++) {
		const JoyBinding binding = mapping.bindings[i];
		if (binding.inputType == TYPE_BUTTON && binding.input.button == p_button) {
			event.type = binding.outputType;
			switch (binding.outputType) {
				case TYPE_BUTTON:
					event.index = (int)binding.output.button;
					return event;
				case TYPE_AXIS:
					event.index = (int)binding.output.axis.axis;
					switch (binding.output.axis.range) {
						case POSITIVE_HALF_AXIS:
							event.value = 1;
							break;
						case NEGATIVE_HALF_AXIS:
							event.value = -1;
							break;
						case FULL_AXIS:
							// It doesn't make sense for a button to map to a full axis,
							// but keeping as a default for a trigger with a positive half-axis.
							event.value = 1;
							break;
					}
					return event;
				default:
					ERR_PRINT_ONCE("Joypad button mapping error.");
			}
		}
	}
	return event;
}

Input::JoyEvent Input::_get_mapped_axis_event(const JoyDeviceMapping &mapping, JoyAxis p_axis, float p_value, JoyAxisRange &r_range) {
	JoyEvent event;

	for (int i = 0; i < mapping.bindings.size(); i++) {
		const JoyBinding binding = mapping.bindings[i];
		if (binding.inputType == TYPE_AXIS && binding.input.axis.axis == p_axis) {
			float value = p_value;
			if (binding.input.axis.invert) {
				value = -value;
			}
			if (binding.input.axis.range == FULL_AXIS ||
					(binding.input.axis.range == POSITIVE_HALF_AXIS && value >= 0) ||
					(binding.input.axis.range == NEGATIVE_HALF_AXIS && value < 0)) {
				event.type = binding.outputType;
				float shifted_positive_value = 0;
				switch (binding.input.axis.range) {
					case POSITIVE_HALF_AXIS:
						shifted_positive_value = value;
						break;
					case NEGATIVE_HALF_AXIS:
						shifted_positive_value = value + 1;
						break;
					case FULL_AXIS:
						shifted_positive_value = (value + 1) / 2;
						break;
				}
				switch (binding.outputType) {
					case TYPE_BUTTON:
						event.index = (int)binding.output.button;
						switch (binding.input.axis.range) {
							case POSITIVE_HALF_AXIS:
								event.value = shifted_positive_value;
								break;
							case NEGATIVE_HALF_AXIS:
								event.value = 1 - shifted_positive_value;
								break;
							case FULL_AXIS:
								// It doesn't make sense for a full axis to map to a button,
								// but keeping as a default for a trigger with a positive half-axis.
								event.value = (shifted_positive_value * 2) - 1;
								break;
						}
						return event;
					case TYPE_AXIS:
						event.index = (int)binding.output.axis.axis;
						event.value = value;
						r_range = binding.output.axis.range;
						if (binding.output.axis.range != binding.input.axis.range) {
							switch (binding.output.axis.range) {
								case POSITIVE_HALF_AXIS:
									event.value = shifted_positive_value;
									break;
								case NEGATIVE_HALF_AXIS:
									event.value = shifted_positive_value - 1;
									break;
								case FULL_AXIS:
									event.value = (shifted_positive_value * 2) - 1;
									break;
							}
						}
						return event;
					default:
						ERR_PRINT_ONCE("Joypad axis mapping error.");
				}
			}
		}
	}
	return event;
}

void Input::_get_mapped_hat_events(const JoyDeviceMapping &mapping, HatDir p_hat, JoyEvent r_events[(size_t)HatDir::MAX]) {
	for (int i = 0; i < mapping.bindings.size(); i++) {
		const JoyBinding binding = mapping.bindings[i];
		if (binding.inputType == TYPE_HAT && binding.input.hat.hat == p_hat) {
			HatDir hat_direction;
			switch (binding.input.hat.hat_mask) {
				case HatMask::UP:
					hat_direction = HatDir::UP;
					break;
				case HatMask::RIGHT:
					hat_direction = HatDir::RIGHT;
					break;
				case HatMask::DOWN:
					hat_direction = HatDir::DOWN;
					break;
				case HatMask::LEFT:
					hat_direction = HatDir::LEFT;
					break;
				default:
					ERR_PRINT_ONCE("Joypad button mapping error.");
					continue;
			}

			r_events[(size_t)hat_direction].type = binding.outputType;
			switch (binding.outputType) {
				case TYPE_BUTTON:
					r_events[(size_t)hat_direction].index = (int)binding.output.button;
					break;
				case TYPE_AXIS:
					r_events[(size_t)hat_direction].index = (int)binding.output.axis.axis;
					switch (binding.output.axis.range) {
						case POSITIVE_HALF_AXIS:
							r_events[(size_t)hat_direction].value = 1;
							break;
						case NEGATIVE_HALF_AXIS:
							r_events[(size_t)hat_direction].value = -1;
							break;
						case FULL_AXIS:
							// It doesn't make sense for a hat direction to map to a full axis,
							// but keeping as a default for a trigger with a positive half-axis.
							r_events[(size_t)hat_direction].value = 1;
							break;
					}
					break;
				default:
					ERR_PRINT_ONCE("Joypad button mapping error.");
			}
		}
	}
}

JoyButton Input::_get_output_button(String output) {
	for (int i = 0; i < (int)JoyButton::SDL_MAX; i++) {
		if (output == _joy_buttons[i]) {
			return JoyButton(i);
		}
	}
	return JoyButton::INVALID;
}

JoyAxis Input::_get_output_axis(String output) {
	for (int i = 0; i < (int)JoyAxis::SDL_MAX; i++) {
		if (output == _joy_axes[i]) {
			return JoyAxis(i);
		}
	}
	return JoyAxis::INVALID;
}

void Input::parse_mapping(String p_mapping) {
	_THREAD_SAFE_METHOD_;
	JoyDeviceMapping mapping;

	Vector<String> entry = p_mapping.split(",");
	if (entry.size() < 2) {
		return;
	}

	CharString uid;
	uid.resize(17);

	mapping.uid = entry[0];
	mapping.name = entry[1];

	int idx = 1;
	while (++idx < entry.size()) {
		if (entry[idx].is_empty()) {
			continue;
		}

		String output = entry[idx].get_slice(":", 0).replace(" ", "");
		String input = entry[idx].get_slice(":", 1).replace(" ", "");
		if (output.length() < 1 || input.length() < 2) {
			continue;
		}

		if (output == "platform" || output == "hint") {
			continue;
		}

		JoyAxisRange output_range = FULL_AXIS;
		if (output[0] == '+' || output[0] == '-') {
			ERR_CONTINUE_MSG(output.length() < 2,
					vformat("Invalid output entry \"%s\" in mapping:\n%s", entry[idx], p_mapping));
			if (output[0] == '+') {
				output_range = POSITIVE_HALF_AXIS;
			} else if (output[0] == '-') {
				output_range = NEGATIVE_HALF_AXIS;
			}
			output = output.substr(1);
		}

		JoyAxisRange input_range = FULL_AXIS;
		if (input[0] == '+') {
			input_range = POSITIVE_HALF_AXIS;
			input = input.substr(1);
		} else if (input[0] == '-') {
			input_range = NEGATIVE_HALF_AXIS;
			input = input.substr(1);
		}
		bool invert_axis = false;
		if (input[input.length() - 1] == '~') {
			invert_axis = true;
			input = input.left(input.length() - 1);
		}

		JoyButton output_button = _get_output_button(output);
		JoyAxis output_axis = _get_output_axis(output);
		if (output_button == JoyButton::INVALID && output_axis == JoyAxis::INVALID) {
			print_verbose(vformat("Unrecognized output string \"%s\" in mapping:\n%s", output, p_mapping));
		}
		ERR_CONTINUE_MSG(output_button != JoyButton::INVALID && output_axis != JoyAxis::INVALID,
				vformat("Output string \"%s\" matched both button and axis in mapping:\n%s", output, p_mapping));

		JoyBinding binding;
		if (output_button != JoyButton::INVALID) {
			binding.outputType = TYPE_BUTTON;
			binding.output.button = output_button;
		} else if (output_axis != JoyAxis::INVALID) {
			binding.outputType = TYPE_AXIS;
			binding.output.axis.axis = output_axis;
			binding.output.axis.range = output_range;
		}

		switch (input[0]) {
			case 'b':
				binding.inputType = TYPE_BUTTON;
				binding.input.button = (JoyButton)input.substr(1).to_int();
				break;
			case 'a':
				binding.inputType = TYPE_AXIS;
				binding.input.axis.axis = (JoyAxis)input.substr(1).to_int();
				binding.input.axis.range = input_range;
				binding.input.axis.invert = invert_axis;
				break;
			case 'h':
				ERR_CONTINUE_MSG(input.length() != 4 || input[2] != '.',
						vformat("Invalid had input \"%s\" in mapping:\n%s", input, p_mapping));
				binding.inputType = TYPE_HAT;
				binding.input.hat.hat = (HatDir)input.substr(1, 1).to_int();
				binding.input.hat.hat_mask = static_cast<HatMask>(input.substr(3).to_int());
				break;
			default:
				ERR_CONTINUE_MSG(true, vformat("Unrecognized input string \"%s\" in mapping:\n%s", input, p_mapping));
		}

		mapping.bindings.push_back(binding);
	}

	map_db.push_back(mapping);
}

void Input::add_joy_mapping(String p_mapping, bool p_update_existing) {
	parse_mapping(p_mapping);
	if (p_update_existing) {
		Vector<String> entry = p_mapping.split(",");
		String uid = entry[0];
		for (KeyValue<int, Joypad> &E : joy_names) {
			Joypad &joy = E.value;
			if (joy.uid == uid) {
				joy.mapping = map_db.size() - 1;
			}
		}
	}
}

void Input::remove_joy_mapping(String p_guid) {
	for (int i = map_db.size() - 1; i >= 0; i--) {
		if (p_guid == map_db[i].uid) {
			map_db.remove_at(i);
		}
	}
	for (KeyValue<int, Joypad> &E : joy_names) {
		Joypad &joy = E.value;
		if (joy.uid == p_guid) {
			joy.mapping = -1;
		}
	}
}

void Input::set_fallback_mapping(String p_guid) {
	for (int i = 0; i < map_db.size(); i++) {
		if (map_db[i].uid == p_guid) {
			fallback_mapping = i;
			return;
		}
	}
}

//platforms that use the remapping system can override and call to these ones
bool Input::is_joy_known(int p_device) {
	if (joy_names.has(p_device)) {
		int mapping = joy_names[p_device].mapping;
		if (mapping != -1 && mapping != fallback_mapping) {
			return true;
		}
	}
	return false;
}

String Input::get_joy_guid(int p_device) const {
	ERR_FAIL_COND_V(!joy_names.has(p_device), "");
	return joy_names[p_device].uid;
}

Dictionary Input::get_joy_info(int p_device) const {
	ERR_FAIL_COND_V(!joy_names.has(p_device), Dictionary());
	return joy_names[p_device].info;
}

bool Input::should_ignore_device(int p_vendor_id, int p_product_id) const {
	uint32_t full_id = (((uint32_t)p_vendor_id) << 16) | ((uint16_t)p_product_id);
	return ignored_device_ids.has(full_id);
}

TypedArray<int> Input::get_connected_joypads() {
	TypedArray<int> ret;
	HashMap<int, Joypad>::Iterator elem = joy_names.begin();
	while (elem) {
		if (elem->value.connected) {
			ret.push_back(elem->key);
		}
		++elem;
	}
	return ret;
}

int Input::get_unused_joy_id() {
	for (int i = 0; i < JOYPADS_MAX; i++) {
		if (!joy_names.has(i) || !joy_names[i].connected) {
			return i;
		}
	}
	return -1;
}

Input::Input() {
	singleton = this;

	// Parse default mappings.
	{
		int i = 0;
		while (DefaultControllerMappings::mappings[i]) {
			parse_mapping(DefaultControllerMappings::mappings[i++]);
		}
	}

	// If defined, parse SDL_GAMECONTROLLERCONFIG for possible new mappings/overrides.
	String env_mapping = OS::get_singleton()->get_environment("SDL_GAMECONTROLLERCONFIG");
	if (!env_mapping.is_empty()) {
		Vector<String> entries = env_mapping.split("\n");
		for (int i = 0; i < entries.size(); i++) {
			if (entries[i].is_empty()) {
				continue;
			}
			parse_mapping(entries[i]);
		}
	}

	String env_ignore_devices = OS::get_singleton()->get_environment("SDL_GAMECONTROLLER_IGNORE_DEVICES");
	if (!env_ignore_devices.is_empty()) {
		Vector<String> entries = env_ignore_devices.split(",");
		for (int i = 0; i < entries.size(); i++) {
			Vector<String> vid_pid = entries[i].split("/");

			if (vid_pid.size() < 2) {
				continue;
			}

			print_verbose(vformat("Device Ignored -- Vendor: %s Product: %s", vid_pid[0], vid_pid[1]));
			const uint16_t vid_unswapped = vid_pid[0].hex_to_int();
			const uint16_t pid_unswapped = vid_pid[1].hex_to_int();
			const uint16_t vid = BSWAP16(vid_unswapped);
			const uint16_t pid = BSWAP16(pid_unswapped);

			uint32_t full_id = (((uint32_t)vid) << 16) | ((uint16_t)pid);
			ignored_device_ids.insert(full_id);
		}
	}

	legacy_just_pressed_behavior = GLOBAL_DEF("input_devices/compatibility/legacy_just_pressed_behavior", false);
	if (Engine::get_singleton()->is_editor_hint()) {
		// Always use standard behavior in the editor.
		legacy_just_pressed_behavior = false;
	}
}

Input::~Input() {
	singleton = nullptr;
}

//////////////////////////////////////////////////////////
