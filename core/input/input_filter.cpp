/*************************************************************************/
/*  input_filter.cpp                                                     */
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

#include "input_filter.h"

#include "core/input/default_controller_mappings.h"
#include "core/input/input_map.h"
#include "core/os/os.h"
#include "core/project_settings.h"

#ifdef TOOLS_ENABLED
#include "editor/editor_settings.h"
#endif

InputFilter *InputFilter::singleton = nullptr;

void (*InputFilter::set_mouse_mode_func)(InputFilter::MouseMode) = nullptr;
InputFilter::MouseMode (*InputFilter::get_mouse_mode_func)() = nullptr;
void (*InputFilter::warp_mouse_func)(const Vector2 &p_to_pos) = nullptr;
InputFilter::CursorShape (*InputFilter::get_current_cursor_shape_func)() = nullptr;
void (*InputFilter::set_custom_mouse_cursor_func)(const RES &, InputFilter::CursorShape, const Vector2 &) = nullptr;

InputFilter *InputFilter::get_singleton() {

	return singleton;
}

void InputFilter::set_mouse_mode(MouseMode p_mode) {
	ERR_FAIL_INDEX((int)p_mode, 4);
	set_mouse_mode_func(p_mode);
}

InputFilter::MouseMode InputFilter::get_mouse_mode() const {

	return get_mouse_mode_func();
}

void InputFilter::_bind_methods() {

	ClassDB::bind_method(D_METHOD("is_key_pressed", "keycode"), &InputFilter::is_key_pressed);
	ClassDB::bind_method(D_METHOD("is_mouse_button_pressed", "button"), &InputFilter::is_mouse_button_pressed);
	ClassDB::bind_method(D_METHOD("is_joy_button_pressed", "device", "button"), &InputFilter::is_joy_button_pressed);
	ClassDB::bind_method(D_METHOD("is_action_pressed", "action"), &InputFilter::is_action_pressed);
	ClassDB::bind_method(D_METHOD("is_action_just_pressed", "action"), &InputFilter::is_action_just_pressed);
	ClassDB::bind_method(D_METHOD("is_action_just_released", "action"), &InputFilter::is_action_just_released);
	ClassDB::bind_method(D_METHOD("get_action_strength", "action"), &InputFilter::get_action_strength);
	ClassDB::bind_method(D_METHOD("add_joy_mapping", "mapping", "update_existing"), &InputFilter::add_joy_mapping, DEFVAL(false));
	ClassDB::bind_method(D_METHOD("remove_joy_mapping", "guid"), &InputFilter::remove_joy_mapping);
	ClassDB::bind_method(D_METHOD("joy_connection_changed", "device", "connected", "name", "guid"), &InputFilter::joy_connection_changed);
	ClassDB::bind_method(D_METHOD("is_joy_known", "device"), &InputFilter::is_joy_known);
	ClassDB::bind_method(D_METHOD("get_joy_axis", "device", "axis"), &InputFilter::get_joy_axis);
	ClassDB::bind_method(D_METHOD("get_joy_name", "device"), &InputFilter::get_joy_name);
	ClassDB::bind_method(D_METHOD("get_joy_guid", "device"), &InputFilter::get_joy_guid);
	ClassDB::bind_method(D_METHOD("get_connected_joypads"), &InputFilter::get_connected_joypads);
	ClassDB::bind_method(D_METHOD("get_joy_vibration_strength", "device"), &InputFilter::get_joy_vibration_strength);
	ClassDB::bind_method(D_METHOD("get_joy_vibration_duration", "device"), &InputFilter::get_joy_vibration_duration);
	ClassDB::bind_method(D_METHOD("get_joy_button_string", "button_index"), &InputFilter::get_joy_button_string);
	ClassDB::bind_method(D_METHOD("get_joy_button_index_from_string", "button"), &InputFilter::get_joy_button_index_from_string);
	ClassDB::bind_method(D_METHOD("get_joy_axis_string", "axis_index"), &InputFilter::get_joy_axis_string);
	ClassDB::bind_method(D_METHOD("get_joy_axis_index_from_string", "axis"), &InputFilter::get_joy_axis_index_from_string);
	ClassDB::bind_method(D_METHOD("start_joy_vibration", "device", "weak_magnitude", "strong_magnitude", "duration"), &InputFilter::start_joy_vibration, DEFVAL(0));
	ClassDB::bind_method(D_METHOD("stop_joy_vibration", "device"), &InputFilter::stop_joy_vibration);
	ClassDB::bind_method(D_METHOD("vibrate_handheld", "duration_ms"), &InputFilter::vibrate_handheld, DEFVAL(500));
	ClassDB::bind_method(D_METHOD("get_gravity"), &InputFilter::get_gravity);
	ClassDB::bind_method(D_METHOD("get_accelerometer"), &InputFilter::get_accelerometer);
	ClassDB::bind_method(D_METHOD("get_magnetometer"), &InputFilter::get_magnetometer);
	ClassDB::bind_method(D_METHOD("get_gyroscope"), &InputFilter::get_gyroscope);
	ClassDB::bind_method(D_METHOD("get_last_mouse_speed"), &InputFilter::get_last_mouse_speed);
	ClassDB::bind_method(D_METHOD("get_mouse_button_mask"), &InputFilter::get_mouse_button_mask);
	ClassDB::bind_method(D_METHOD("set_mouse_mode", "mode"), &InputFilter::set_mouse_mode);
	ClassDB::bind_method(D_METHOD("get_mouse_mode"), &InputFilter::get_mouse_mode);
	ClassDB::bind_method(D_METHOD("warp_mouse_position", "to"), &InputFilter::warp_mouse_position);
	ClassDB::bind_method(D_METHOD("action_press", "action", "strength"), &InputFilter::action_press, DEFVAL(1.f));
	ClassDB::bind_method(D_METHOD("action_release", "action"), &InputFilter::action_release);
	ClassDB::bind_method(D_METHOD("set_default_cursor_shape", "shape"), &InputFilter::set_default_cursor_shape, DEFVAL(CURSOR_ARROW));
	ClassDB::bind_method(D_METHOD("get_current_cursor_shape"), &InputFilter::get_current_cursor_shape);
	ClassDB::bind_method(D_METHOD("set_custom_mouse_cursor", "image", "shape", "hotspot"), &InputFilter::set_custom_mouse_cursor, DEFVAL(CURSOR_ARROW), DEFVAL(Vector2()));
	ClassDB::bind_method(D_METHOD("parse_input_event", "event"), &InputFilter::parse_input_event);
	ClassDB::bind_method(D_METHOD("set_use_accumulated_input", "enable"), &InputFilter::set_use_accumulated_input);

	BIND_ENUM_CONSTANT(MOUSE_MODE_VISIBLE);
	BIND_ENUM_CONSTANT(MOUSE_MODE_HIDDEN);
	BIND_ENUM_CONSTANT(MOUSE_MODE_CAPTURED);
	BIND_ENUM_CONSTANT(MOUSE_MODE_CONFINED);

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

void InputFilter::get_argument_options(const StringName &p_function, int p_idx, List<String> *r_options) const {
#ifdef TOOLS_ENABLED

	const String quote_style = EDITOR_DEF("text_editor/completion/use_single_quotes", 0) ? "'" : "\"";

	String pf = p_function;
	if (p_idx == 0 && (pf == "is_action_pressed" || pf == "action_press" || pf == "action_release" || pf == "is_action_just_pressed" || pf == "is_action_just_released" || pf == "get_action_strength")) {

		List<PropertyInfo> pinfo;
		ProjectSettings::get_singleton()->get_property_list(&pinfo);

		for (List<PropertyInfo>::Element *E = pinfo.front(); E; E = E->next()) {
			const PropertyInfo &pi = E->get();

			if (!pi.name.begins_with("input/"))
				continue;

			String name = pi.name.substr(pi.name.find("/") + 1, pi.name.length());
			r_options->push_back(quote_style + name + quote_style);
		}
	}
#endif
}

void InputFilter::SpeedTrack::update(const Vector2 &p_delta_p) {

	uint64_t tick = OS::get_singleton()->get_ticks_usec();
	uint32_t tdiff = tick - last_tick;
	float delta_t = tdiff / 1000000.0;
	last_tick = tick;

	accum += p_delta_p;
	accum_t += delta_t;

	if (accum_t > max_ref_frame * 10)
		accum_t = max_ref_frame * 10;

	while (accum_t >= min_ref_frame) {

		float slice_t = min_ref_frame / accum_t;
		Vector2 slice = accum * slice_t;
		accum = accum - slice;
		accum_t -= min_ref_frame;

		speed = (slice / min_ref_frame).linear_interpolate(speed, min_ref_frame / max_ref_frame);
	}
}

void InputFilter::SpeedTrack::reset() {
	last_tick = OS::get_singleton()->get_ticks_usec();
	speed = Vector2();
	accum_t = 0;
}

InputFilter::SpeedTrack::SpeedTrack() {

	min_ref_frame = 0.1;
	max_ref_frame = 0.3;
	reset();
}

bool InputFilter::is_key_pressed(int p_keycode) const {

	_THREAD_SAFE_METHOD_
	return keys_pressed.has(p_keycode);
}

bool InputFilter::is_mouse_button_pressed(int p_button) const {

	_THREAD_SAFE_METHOD_
	return (mouse_button_mask & (1 << (p_button - 1))) != 0;
}

static int _combine_device(int p_value, int p_device) {

	return p_value | (p_device << 20);
}

bool InputFilter::is_joy_button_pressed(int p_device, int p_button) const {

	_THREAD_SAFE_METHOD_
	return joy_buttons_pressed.has(_combine_device(p_button, p_device));
}

bool InputFilter::is_action_pressed(const StringName &p_action) const {

	return action_state.has(p_action) && action_state[p_action].pressed;
}

bool InputFilter::is_action_just_pressed(const StringName &p_action) const {

	const Map<StringName, Action>::Element *E = action_state.find(p_action);
	if (!E)
		return false;

	if (Engine::get_singleton()->is_in_physics_frame()) {
		return E->get().pressed && E->get().physics_frame == Engine::get_singleton()->get_physics_frames();
	} else {
		return E->get().pressed && E->get().idle_frame == Engine::get_singleton()->get_idle_frames();
	}
}

bool InputFilter::is_action_just_released(const StringName &p_action) const {

	const Map<StringName, Action>::Element *E = action_state.find(p_action);
	if (!E)
		return false;

	if (Engine::get_singleton()->is_in_physics_frame()) {
		return !E->get().pressed && E->get().physics_frame == Engine::get_singleton()->get_physics_frames();
	} else {
		return !E->get().pressed && E->get().idle_frame == Engine::get_singleton()->get_idle_frames();
	}
}

float InputFilter::get_action_strength(const StringName &p_action) const {
	const Map<StringName, Action>::Element *E = action_state.find(p_action);
	if (!E)
		return 0.0f;

	return E->get().strength;
}

float InputFilter::get_joy_axis(int p_device, int p_axis) const {

	_THREAD_SAFE_METHOD_
	int c = _combine_device(p_axis, p_device);
	if (_joy_axis.has(c)) {
		return _joy_axis[c];
	} else {
		return 0;
	}
}

String InputFilter::get_joy_name(int p_idx) {

	_THREAD_SAFE_METHOD_
	return joy_names[p_idx].name;
};

Vector2 InputFilter::get_joy_vibration_strength(int p_device) {
	if (joy_vibration.has(p_device)) {
		return Vector2(joy_vibration[p_device].weak_magnitude, joy_vibration[p_device].strong_magnitude);
	} else {
		return Vector2(0, 0);
	}
}

uint64_t InputFilter::get_joy_vibration_timestamp(int p_device) {
	if (joy_vibration.has(p_device)) {
		return joy_vibration[p_device].timestamp;
	} else {
		return 0;
	}
}

float InputFilter::get_joy_vibration_duration(int p_device) {
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
};

void InputFilter::joy_connection_changed(int p_idx, bool p_connected, String p_name, String p_guid) {

	_THREAD_SAFE_METHOD_
	Joypad js;
	js.name = p_connected ? p_name : "";
	js.uid = p_connected ? p_guid : "";

	if (p_connected) {

		String uidname = p_guid;
		if (p_guid == "") {
			int uidlen = MIN(p_name.length(), 16);
			for (int i = 0; i < uidlen; i++) {
				uidname = uidname + _hex_str(p_name[i]);
			};
		};
		js.uid = uidname;
		js.connected = true;
		int mapping = fallback_mapping;
		for (int i = 0; i < map_db.size(); i++) {
			if (js.uid == map_db[i].uid) {
				mapping = i;
				js.name = map_db[i].name;
			};
		};
		js.mapping = mapping;
	} else {
		js.connected = false;
		for (int i = 0; i < JOY_BUTTON_MAX; i++) {

			if (i < JOY_AXIS_MAX)
				set_joy_axis(p_idx, i, 0.0f);

			int c = _combine_device(i, p_idx);
			joy_buttons_pressed.erase(c);
		};
	};
	joy_names[p_idx] = js;

	emit_signal("joy_connection_changed", p_idx, p_connected);
};

Vector3 InputFilter::get_gravity() const {

	_THREAD_SAFE_METHOD_
	return gravity;
}

Vector3 InputFilter::get_accelerometer() const {

	_THREAD_SAFE_METHOD_
	return accelerometer;
}

Vector3 InputFilter::get_magnetometer() const {

	_THREAD_SAFE_METHOD_
	return magnetometer;
}

Vector3 InputFilter::get_gyroscope() const {

	_THREAD_SAFE_METHOD_
	return gyroscope;
}

void InputFilter::parse_input_event(const Ref<InputEvent> &p_event) {

	_parse_input_event_impl(p_event, false);
}

void InputFilter::_parse_input_event_impl(const Ref<InputEvent> &p_event, bool p_is_emulated) {

	// Notes on mouse-touch emulation:
	// - Emulated mouse events are parsed, that is, re-routed to this method, so they make the same effects
	//   as true mouse events. The only difference is the situation is flagged as emulated so they are not
	//   emulated back to touch events in an endless loop.
	// - Emulated touch events are handed right to the main loop (i.e., the SceneTree) because they don't
	//   require additional handling by this class.

	_THREAD_SAFE_METHOD_

	Ref<InputEventKey> k = p_event;
	if (k.is_valid() && !k->is_echo() && k->get_keycode() != 0) {
		if (k->is_pressed())
			keys_pressed.insert(k->get_keycode());
		else
			keys_pressed.erase(k->get_keycode());
	}

	Ref<InputEventMouseButton> mb = p_event;

	if (mb.is_valid()) {

		if (mb->is_pressed()) {
			mouse_button_mask |= (1 << (mb->get_button_index() - 1));
		} else {
			mouse_button_mask &= ~(1 << (mb->get_button_index() - 1));
		}

		Point2 pos = mb->get_global_position();
		if (mouse_pos != pos) {
			set_mouse_position(pos);
		}

		if (event_dispatch_function && emulate_touch_from_mouse && !p_is_emulated && mb->get_button_index() == 1) {
			Ref<InputEventScreenTouch> touch_event;
			touch_event.instance();
			touch_event->set_pressed(mb->is_pressed());
			touch_event->set_position(mb->get_position());
			event_dispatch_function(touch_event);
		}
	}

	Ref<InputEventMouseMotion> mm = p_event;

	if (mm.is_valid()) {

		Point2 pos = mm->get_global_position();
		if (mouse_pos != pos) {
			set_mouse_position(pos);
		}

		if (event_dispatch_function && emulate_touch_from_mouse && !p_is_emulated && mm->get_button_mask() & 1) {
			Ref<InputEventScreenDrag> drag_event;
			drag_event.instance();

			drag_event->set_position(mm->get_position());
			drag_event->set_relative(mm->get_relative());
			drag_event->set_speed(mm->get_speed());

			event_dispatch_function(drag_event);
		}
	}

	Ref<InputEventScreenTouch> st = p_event;

	if (st.is_valid()) {

		if (st->is_pressed()) {
			SpeedTrack &track = touch_speed_track[st->get_index()];
			track.reset();
		} else {
			// Since a pointer index may not occur again (OSs may or may not reuse them),
			// imperatively remove it from the map to keep no fossil entries in it
			touch_speed_track.erase(st->get_index());
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
				button_event.instance();

				button_event->set_device(InputEvent::DEVICE_ID_TOUCH_MOUSE);
				button_event->set_position(st->get_position());
				button_event->set_global_position(st->get_position());
				button_event->set_pressed(st->is_pressed());
				button_event->set_button_index(BUTTON_LEFT);
				if (st->is_pressed()) {
					button_event->set_button_mask(mouse_button_mask | (1 << (BUTTON_LEFT - 1)));
				} else {
					button_event->set_button_mask(mouse_button_mask & ~(1 << (BUTTON_LEFT - 1)));
				}

				_parse_input_event_impl(button_event, true);
			}
		}
	}

	Ref<InputEventScreenDrag> sd = p_event;

	if (sd.is_valid()) {

		SpeedTrack &track = touch_speed_track[sd->get_index()];
		track.update(sd->get_relative());
		sd->set_speed(track.speed);

		if (emulate_mouse_from_touch && sd->get_index() == mouse_from_touch_index) {

			Ref<InputEventMouseMotion> motion_event;
			motion_event.instance();

			motion_event->set_device(InputEvent::DEVICE_ID_TOUCH_MOUSE);
			motion_event->set_position(sd->get_position());
			motion_event->set_global_position(sd->get_position());
			motion_event->set_relative(sd->get_relative());
			motion_event->set_speed(sd->get_speed());
			motion_event->set_button_mask(mouse_button_mask);

			_parse_input_event_impl(motion_event, true);
		}
	}

	Ref<InputEventJoypadButton> jb = p_event;

	if (jb.is_valid()) {

		int c = _combine_device(jb->get_button_index(), jb->get_device());

		if (jb->is_pressed())
			joy_buttons_pressed.insert(c);
		else
			joy_buttons_pressed.erase(c);
	}

	Ref<InputEventJoypadMotion> jm = p_event;

	if (jm.is_valid()) {
		set_joy_axis(jm->get_device(), jm->get_axis(), jm->get_axis_value());
	}

	Ref<InputEventGesture> ge = p_event;

	if (ge.is_valid()) {

		if (event_dispatch_function) {
			event_dispatch_function(ge);
		}
	}

	for (const Map<StringName, InputMap::Action>::Element *E = InputMap::get_singleton()->get_action_map().front(); E; E = E->next()) {
		if (InputMap::get_singleton()->event_is_action(p_event, E->key())) {

			// Save the action's state
			if (!p_event->is_echo() && is_action_pressed(E->key()) != p_event->is_action_pressed(E->key())) {
				Action action;
				action.physics_frame = Engine::get_singleton()->get_physics_frames();
				action.idle_frame = Engine::get_singleton()->get_idle_frames();
				action.pressed = p_event->is_action_pressed(E->key());
				action.strength = 0.f;
				action_state[E->key()] = action;
			}
			action_state[E->key()].strength = p_event->get_action_strength(E->key());
		}
	}

	if (event_dispatch_function)
		event_dispatch_function(p_event);
}

void InputFilter::set_joy_axis(int p_device, int p_axis, float p_value) {

	_THREAD_SAFE_METHOD_
	int c = _combine_device(p_axis, p_device);
	_joy_axis[c] = p_value;
}

void InputFilter::start_joy_vibration(int p_device, float p_weak_magnitude, float p_strong_magnitude, float p_duration) {
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

void InputFilter::stop_joy_vibration(int p_device) {
	_THREAD_SAFE_METHOD_
	VibrationInfo vibration;
	vibration.weak_magnitude = 0;
	vibration.strong_magnitude = 0;
	vibration.duration = 0;
	vibration.timestamp = OS::get_singleton()->get_ticks_usec();
	joy_vibration[p_device] = vibration;
}

void InputFilter::vibrate_handheld(int p_duration_ms) {
	OS::get_singleton()->vibrate_handheld(p_duration_ms);
}

void InputFilter::set_gravity(const Vector3 &p_gravity) {

	_THREAD_SAFE_METHOD_

	gravity = p_gravity;
}

void InputFilter::set_accelerometer(const Vector3 &p_accel) {

	_THREAD_SAFE_METHOD_

	accelerometer = p_accel;
}

void InputFilter::set_magnetometer(const Vector3 &p_magnetometer) {

	_THREAD_SAFE_METHOD_

	magnetometer = p_magnetometer;
}

void InputFilter::set_gyroscope(const Vector3 &p_gyroscope) {

	_THREAD_SAFE_METHOD_

	gyroscope = p_gyroscope;
}

void InputFilter::set_mouse_position(const Point2 &p_posf) {

	mouse_speed_track.update(p_posf - mouse_pos);
	mouse_pos = p_posf;
}

Point2 InputFilter::get_mouse_position() const {

	return mouse_pos;
}
Point2 InputFilter::get_last_mouse_speed() const {

	return mouse_speed_track.speed;
}

int InputFilter::get_mouse_button_mask() const {

	return mouse_button_mask; // do not trust OS implementation, should remove it - OS::get_singleton()->get_mouse_button_state();
}

void InputFilter::warp_mouse_position(const Vector2 &p_to) {
	warp_mouse_func(p_to);
}

Point2i InputFilter::warp_mouse_motion(const Ref<InputEventMouseMotion> &p_motion, const Rect2 &p_rect) {

	// The relative distance reported for the next event after a warp is in the boundaries of the
	// size of the rect on that axis, but it may be greater, in which case there's not problem as fmod()
	// will warp it, but if the pointer has moved in the opposite direction between the pointer relocation
	// and the subsequent event, the reported relative distance will be less than the size of the rect
	// and thus fmod() will be disabled for handling the situation.
	// And due to this mouse warping mechanism being stateless, we need to apply some heuristics to
	// detect the warp: if the relative distance is greater than the half of the size of the relevant rect
	// (checked per each axis), it will be considered as the consequence of a former pointer warp.

	const Point2i rel_sgn(p_motion->get_relative().x >= 0.0f ? 1 : -1, p_motion->get_relative().y >= 0.0 ? 1 : -1);
	const Size2i warp_margin = p_rect.size * 0.5f;
	const Point2i rel_warped(
			Math::fmod(p_motion->get_relative().x + rel_sgn.x * warp_margin.x, p_rect.size.x) - rel_sgn.x * warp_margin.x,
			Math::fmod(p_motion->get_relative().y + rel_sgn.y * warp_margin.y, p_rect.size.y) - rel_sgn.y * warp_margin.y);

	const Point2i pos_local = p_motion->get_global_position() - p_rect.position;
	const Point2i pos_warped(Math::fposmod(pos_local.x, p_rect.size.x), Math::fposmod(pos_local.y, p_rect.size.y));
	if (pos_warped != pos_local) {
		warp_mouse_position(pos_warped + p_rect.position);
	}

	return rel_warped;
}

void InputFilter::iteration(float p_step) {
}

void InputFilter::action_press(const StringName &p_action, float p_strength) {

	Action action;

	action.physics_frame = Engine::get_singleton()->get_physics_frames();
	action.idle_frame = Engine::get_singleton()->get_idle_frames();
	action.pressed = true;
	action.strength = p_strength;

	action_state[p_action] = action;
}

void InputFilter::action_release(const StringName &p_action) {

	Action action;

	action.physics_frame = Engine::get_singleton()->get_physics_frames();
	action.idle_frame = Engine::get_singleton()->get_idle_frames();
	action.pressed = false;
	action.strength = 0.f;

	action_state[p_action] = action;
}

void InputFilter::set_emulate_touch_from_mouse(bool p_emulate) {

	emulate_touch_from_mouse = p_emulate;
}

bool InputFilter::is_emulating_touch_from_mouse() const {

	return emulate_touch_from_mouse;
}

// Calling this whenever the game window is focused helps unstucking the "touch mouse"
// if the OS or its abstraction class hasn't properly reported that touch pointers raised
void InputFilter::ensure_touch_mouse_raised() {

	if (mouse_from_touch_index != -1) {
		mouse_from_touch_index = -1;

		Ref<InputEventMouseButton> button_event;
		button_event.instance();

		button_event->set_device(InputEvent::DEVICE_ID_TOUCH_MOUSE);
		button_event->set_position(mouse_pos);
		button_event->set_global_position(mouse_pos);
		button_event->set_pressed(false);
		button_event->set_button_index(BUTTON_LEFT);
		button_event->set_button_mask(mouse_button_mask & ~(1 << (BUTTON_LEFT - 1)));

		_parse_input_event_impl(button_event, true);
	}
}

void InputFilter::set_emulate_mouse_from_touch(bool p_emulate) {

	emulate_mouse_from_touch = p_emulate;
}

bool InputFilter::is_emulating_mouse_from_touch() const {

	return emulate_mouse_from_touch;
}

InputFilter::CursorShape InputFilter::get_default_cursor_shape() const {

	return default_shape;
}

void InputFilter::set_default_cursor_shape(CursorShape p_shape) {

	if (default_shape == p_shape)
		return;

	default_shape = p_shape;
	// The default shape is set in Viewport::_gui_input_event. To instantly
	// see the shape in the viewport we need to trigger a mouse motion event.
	Ref<InputEventMouseMotion> mm;
	mm.instance();
	mm->set_position(mouse_pos);
	mm->set_global_position(mouse_pos);
	parse_input_event(mm);
}

InputFilter::CursorShape InputFilter::get_current_cursor_shape() const {

	return get_current_cursor_shape_func();
}

void InputFilter::set_custom_mouse_cursor(const RES &p_cursor, CursorShape p_shape, const Vector2 &p_hotspot) {
	if (Engine::get_singleton()->is_editor_hint())
		return;

	set_custom_mouse_cursor_func(p_cursor, p_shape, p_hotspot);
}

void InputFilter::accumulate_input_event(const Ref<InputEvent> &p_event) {
	ERR_FAIL_COND(p_event.is_null());

	if (!use_accumulated_input) {
		parse_input_event(p_event);
		return;
	}
	if (!accumulated_events.empty() && accumulated_events.back()->get()->accumulate(p_event)) {
		return; //event was accumulated, exit
	}

	accumulated_events.push_back(p_event);
}
void InputFilter::flush_accumulated_events() {

	while (accumulated_events.front()) {
		parse_input_event(accumulated_events.front()->get());
		accumulated_events.pop_front();
	}
}

void InputFilter::set_use_accumulated_input(bool p_enable) {

	use_accumulated_input = p_enable;
}

void InputFilter::release_pressed_events() {

	flush_accumulated_events(); // this is needed to release actions strengths

	keys_pressed.clear();
	joy_buttons_pressed.clear();
	_joy_axis.clear();

	for (Map<StringName, InputFilter::Action>::Element *E = action_state.front(); E; E = E->next()) {
		if (E->get().pressed)
			action_release(E->key());
	}
}

void InputFilter::set_event_dispatch_function(EventDispatchFunc p_function) {
	event_dispatch_function = p_function;
}

void InputFilter::joy_button(int p_device, int p_button, bool p_pressed) {

	_THREAD_SAFE_METHOD_;
	Joypad &joy = joy_names[p_device];
	//printf("got button %i, mapping is %i\n", p_button, joy.mapping);
	if (joy.last_buttons[p_button] == p_pressed) {
		return;
	}
	joy.last_buttons[p_button] = p_pressed;
	if (joy.mapping == -1) {
		_button_event(p_device, p_button, p_pressed);
		return;
	}

	const Map<int, JoyEvent>::Element *el = map_db[joy.mapping].buttons.find(p_button);
	if (!el) {
		//don't process un-mapped events for now, it could mess things up badly for devices with additional buttons/axis
		//return _button_event(p_last_id, p_device, p_button, p_pressed);
		return;
	}

	JoyEvent map = el->get();
	if (map.type == TYPE_BUTTON) {
		//fake additional axis event for triggers
		if (map.index == JOY_L2 || map.index == JOY_R2) {
			float value = p_pressed ? 1.0f : 0.0f;
			int axis = map.index == JOY_L2 ? JOY_ANALOG_L2 : JOY_ANALOG_R2;
			_axis_event(p_device, axis, value);
		}
		_button_event(p_device, map.index, p_pressed);
		return;
	}

	if (map.type == TYPE_AXIS) {
		_axis_event(p_device, map.index, p_pressed ? 1.0 : 0.0);
	}
	// no event?
}

void InputFilter::joy_axis(int p_device, int p_axis, const JoyAxis &p_value) {

	_THREAD_SAFE_METHOD_;

	ERR_FAIL_INDEX(p_axis, JOY_AXIS_MAX);

	Joypad &joy = joy_names[p_device];

	if (joy.last_axis[p_axis] == p_value.value) {
		return;
	}

	if (p_value.value > joy.last_axis[p_axis]) {

		if (p_value.value < joy.last_axis[p_axis] + joy.filter) {

			return;
		}
	} else if (p_value.value > joy.last_axis[p_axis] - joy.filter) {

		return;
	}

	//when changing direction quickly, insert fake event to release pending inputmap actions
	float last = joy.last_axis[p_axis];
	if (p_value.min == 0 && (last < 0.25 || last > 0.75) && (last - 0.5) * (p_value.value - 0.5) < 0) {
		JoyAxis jx;
		jx.min = p_value.min;
		jx.value = p_value.value < 0.5 ? 0.6 : 0.4;
		joy_axis(p_device, p_axis, jx);
	} else if (ABS(last) > 0.5 && last * p_value.value < 0) {
		JoyAxis jx;
		jx.min = p_value.min;
		jx.value = p_value.value < 0 ? 0.1 : -0.1;
		joy_axis(p_device, p_axis, jx);
	}

	joy.last_axis[p_axis] = p_value.value;
	float val = p_value.min == 0 ? -1.0f + 2.0f * p_value.value : p_value.value;

	if (joy.mapping == -1) {
		_axis_event(p_device, p_axis, val);
		return;
	};

	const Map<int, JoyEvent>::Element *el = map_db[joy.mapping].axis.find(p_axis);
	if (!el) {
		//return _axis_event(p_last_id, p_device, p_axis, p_value);
		return;
	};

	JoyEvent map = el->get();

	if (map.type == TYPE_BUTTON) {
		//send axis event for triggers
		if (map.index == JOY_L2 || map.index == JOY_R2) {
			float value = p_value.min == 0 ? p_value.value : 0.5f + p_value.value / 2.0f;
			int axis = map.index == JOY_L2 ? JOY_ANALOG_L2 : JOY_ANALOG_R2;
			_axis_event(p_device, axis, value);
		}

		if (map.index == JOY_DPAD_UP || map.index == JOY_DPAD_DOWN) {
			bool pressed = p_value.value != 0.0f;
			int button = p_value.value < 0 ? JOY_DPAD_UP : JOY_DPAD_DOWN;

			if (!pressed) {
				if (joy_buttons_pressed.has(_combine_device(JOY_DPAD_UP, p_device))) {
					_button_event(p_device, JOY_DPAD_UP, false);
				}
				if (joy_buttons_pressed.has(_combine_device(JOY_DPAD_DOWN, p_device))) {
					_button_event(p_device, JOY_DPAD_DOWN, false);
				}
			}
			if (pressed == joy_buttons_pressed.has(_combine_device(button, p_device))) {
				return;
			}
			_button_event(p_device, button, true);
			return;
		}
		if (map.index == JOY_DPAD_LEFT || map.index == JOY_DPAD_RIGHT) {
			bool pressed = p_value.value != 0.0f;
			int button = p_value.value < 0 ? JOY_DPAD_LEFT : JOY_DPAD_RIGHT;

			if (!pressed) {
				if (joy_buttons_pressed.has(_combine_device(JOY_DPAD_LEFT, p_device))) {
					_button_event(p_device, JOY_DPAD_LEFT, false);
				}
				if (joy_buttons_pressed.has(_combine_device(JOY_DPAD_RIGHT, p_device))) {
					_button_event(p_device, JOY_DPAD_RIGHT, false);
				}
			}
			if (pressed == joy_buttons_pressed.has(_combine_device(button, p_device))) {
				return;
			}
			_button_event(p_device, button, true);
			return;
		}
		float deadzone = p_value.min == 0 ? 0.5f : 0.0f;
		bool pressed = p_value.value > deadzone;
		if (pressed == joy_buttons_pressed.has(_combine_device(map.index, p_device))) {
			// button already pressed or released, this is an axis bounce value
			return;
		}
		_button_event(p_device, map.index, pressed);
		return;
	}

	if (map.type == TYPE_AXIS) {

		_axis_event(p_device, map.index, val);
		return;
	}
	//printf("invalid mapping\n");
}

void InputFilter::joy_hat(int p_device, int p_val) {

	_THREAD_SAFE_METHOD_;
	const Joypad &joy = joy_names[p_device];

	const JoyEvent *map;

	if (joy.mapping == -1) {
		map = hat_map_default;
	} else {
		map = map_db[joy.mapping].hat;
	};

	int cur_val = joy_names[p_device].hat_current;

	if ((p_val & HAT_MASK_UP) != (cur_val & HAT_MASK_UP)) {
		_button_event(p_device, map[HAT_UP].index, p_val & HAT_MASK_UP);
	}

	if ((p_val & HAT_MASK_RIGHT) != (cur_val & HAT_MASK_RIGHT)) {
		_button_event(p_device, map[HAT_RIGHT].index, p_val & HAT_MASK_RIGHT);
	}
	if ((p_val & HAT_MASK_DOWN) != (cur_val & HAT_MASK_DOWN)) {
		_button_event(p_device, map[HAT_DOWN].index, p_val & HAT_MASK_DOWN);
	}
	if ((p_val & HAT_MASK_LEFT) != (cur_val & HAT_MASK_LEFT)) {
		_button_event(p_device, map[HAT_LEFT].index, p_val & HAT_MASK_LEFT);
	}

	joy_names[p_device].hat_current = p_val;
}

void InputFilter::_button_event(int p_device, int p_index, bool p_pressed) {

	Ref<InputEventJoypadButton> ievent;
	ievent.instance();
	ievent->set_device(p_device);
	ievent->set_button_index(p_index);
	ievent->set_pressed(p_pressed);

	parse_input_event(ievent);
}

void InputFilter::_axis_event(int p_device, int p_axis, float p_value) {

	Ref<InputEventJoypadMotion> ievent;
	ievent.instance();
	ievent->set_device(p_device);
	ievent->set_axis(p_axis);
	ievent->set_axis_value(p_value);

	parse_input_event(ievent);
};

InputFilter::JoyEvent InputFilter::_find_to_event(String p_to) {

	// string names of the SDL buttons in the same order as input_event.h godot buttons
	static const char *buttons[] = { "a", "b", "x", "y", "leftshoulder", "rightshoulder", "lefttrigger", "righttrigger", "leftstick", "rightstick", "back", "start", "dpup", "dpdown", "dpleft", "dpright", "guide", nullptr };

	static const char *axis[] = { "leftx", "lefty", "rightx", "righty", nullptr };

	JoyEvent ret;
	ret.type = -1;
	ret.index = 0;

	int i = 0;
	while (buttons[i]) {

		if (p_to == buttons[i]) {
			ret.type = TYPE_BUTTON;
			ret.index = i;
			ret.value = 0;
			return ret;
		};
		++i;
	};

	i = 0;
	while (axis[i]) {

		if (p_to == axis[i]) {
			ret.type = TYPE_AXIS;
			ret.index = i;
			ret.value = 0;
			return ret;
		};
		++i;
	};

	return ret;
};

void InputFilter::parse_mapping(String p_mapping) {

	_THREAD_SAFE_METHOD_;
	JoyDeviceMapping mapping;
	for (int i = 0; i < HAT_MAX; ++i)
		mapping.hat[i].index = 1024 + i;

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

		if (entry[idx] == "")
			continue;

		String from = entry[idx].get_slice(":", 1).replace(" ", "");
		String to = entry[idx].get_slice(":", 0).replace(" ", "");

		JoyEvent to_event = _find_to_event(to);
		if (to_event.type == -1)
			continue;

		String etype = from.substr(0, 1);
		if (etype == "a") {

			int aid = from.substr(1, from.length() - 1).to_int();
			mapping.axis[aid] = to_event;

		} else if (etype == "b") {

			int bid = from.substr(1, from.length() - 1).to_int();
			mapping.buttons[bid] = to_event;

		} else if (etype == "h") {

			int hat_value = from.get_slice(".", 1).to_int();
			switch (hat_value) {
				case 1:
					mapping.hat[HAT_UP] = to_event;
					break;
				case 2:
					mapping.hat[HAT_RIGHT] = to_event;
					break;
				case 4:
					mapping.hat[HAT_DOWN] = to_event;
					break;
				case 8:
					mapping.hat[HAT_LEFT] = to_event;
					break;
			};
		};
	};
	map_db.push_back(mapping);
	//printf("added mapping with uuid %ls\n", mapping.uid.c_str());
};

void InputFilter::add_joy_mapping(String p_mapping, bool p_update_existing) {
	parse_mapping(p_mapping);
	if (p_update_existing) {
		Vector<String> entry = p_mapping.split(",");
		String uid = entry[0];
		for (int i = 0; i < joy_names.size(); i++) {
			if (uid == joy_names[i].uid) {
				joy_names[i].mapping = map_db.size() - 1;
			}
		}
	}
}

void InputFilter::remove_joy_mapping(String p_guid) {
	for (int i = map_db.size() - 1; i >= 0; i--) {
		if (p_guid == map_db[i].uid) {
			map_db.remove(i);
		}
	}
	for (int i = 0; i < joy_names.size(); i++) {
		if (joy_names[i].uid == p_guid) {
			joy_names[i].mapping = -1;
		}
	}
}

void InputFilter::set_fallback_mapping(String p_guid) {

	for (int i = 0; i < map_db.size(); i++) {
		if (map_db[i].uid == p_guid) {
			fallback_mapping = i;
			return;
		}
	}
}

//platforms that use the remapping system can override and call to these ones
bool InputFilter::is_joy_known(int p_device) {
	int mapping = joy_names[p_device].mapping;
	return mapping != -1 ? (mapping != fallback_mapping) : false;
}

String InputFilter::get_joy_guid(int p_device) const {
	ERR_FAIL_COND_V(!joy_names.has(p_device), "");
	return joy_names[p_device].uid;
}

Array InputFilter::get_connected_joypads() {
	Array ret;
	Map<int, Joypad>::Element *elem = joy_names.front();
	while (elem) {
		if (elem->get().connected) {
			ret.push_back(elem->key());
		}
		elem = elem->next();
	}
	return ret;
}

static const char *_buttons[JOY_BUTTON_MAX] = {
	"Face Button Bottom",
	"Face Button Right",
	"Face Button Left",
	"Face Button Top",
	"L",
	"R",
	"L2",
	"R2",
	"L3",
	"R3",
	"Select",
	"Start",
	"DPAD Up",
	"DPAD Down",
	"DPAD Left",
	"DPAD Right"
};

static const char *_axes[JOY_AXIS_MAX] = {
	"Left Stick X",
	"Left Stick Y",
	"Right Stick X",
	"Right Stick Y",
	"",
	"",
	"L2",
	"R2",
	"",
	""
};

String InputFilter::get_joy_button_string(int p_button) {
	ERR_FAIL_INDEX_V(p_button, JOY_BUTTON_MAX, "");
	return _buttons[p_button];
}

int InputFilter::get_joy_button_index_from_string(String p_button) {
	for (int i = 0; i < JOY_BUTTON_MAX; i++) {
		if (p_button == _buttons[i]) {
			return i;
		}
	}
	ERR_FAIL_V(-1);
}

int InputFilter::get_unused_joy_id() {
	for (int i = 0; i < JOYPADS_MAX; i++) {
		if (!joy_names.has(i) || !joy_names[i].connected) {
			return i;
		}
	}
	return -1;
}

String InputFilter::get_joy_axis_string(int p_axis) {
	ERR_FAIL_INDEX_V(p_axis, JOY_AXIS_MAX, "");
	return _axes[p_axis];
}

int InputFilter::get_joy_axis_index_from_string(String p_axis) {
	for (int i = 0; i < JOY_AXIS_MAX; i++) {
		if (p_axis == _axes[i]) {
			return i;
		}
	}
	ERR_FAIL_V(-1);
}

InputFilter::InputFilter() {

	singleton = this;
	use_accumulated_input = true;
	mouse_button_mask = 0;
	mouse_window = 0;
	emulate_touch_from_mouse = false;
	emulate_mouse_from_touch = false;
	mouse_from_touch_index = -1;
	event_dispatch_function = nullptr;
	default_shape = CURSOR_ARROW;

	hat_map_default[HAT_UP].type = TYPE_BUTTON;
	hat_map_default[HAT_UP].index = JOY_DPAD_UP;
	hat_map_default[HAT_UP].value = 0;

	hat_map_default[HAT_RIGHT].type = TYPE_BUTTON;
	hat_map_default[HAT_RIGHT].index = JOY_DPAD_RIGHT;
	hat_map_default[HAT_RIGHT].value = 0;

	hat_map_default[HAT_DOWN].type = TYPE_BUTTON;
	hat_map_default[HAT_DOWN].index = JOY_DPAD_DOWN;
	hat_map_default[HAT_DOWN].value = 0;

	hat_map_default[HAT_LEFT].type = TYPE_BUTTON;
	hat_map_default[HAT_LEFT].index = JOY_DPAD_LEFT;
	hat_map_default[HAT_LEFT].value = 0;

	fallback_mapping = -1;

	// Parse default mappings.
	{
		int i = 0;
		while (DefaultControllerMappings::mappings[i]) {
			parse_mapping(DefaultControllerMappings::mappings[i++]);
		}
	}

	// If defined, parse SDL_GAMECONTROLLERCONFIG for possible new mappings/overrides.
	String env_mapping = OS::get_singleton()->get_environment("SDL_GAMECONTROLLERCONFIG");
	if (env_mapping != "") {
		Vector<String> entries = env_mapping.split("\n");
		for (int i = 0; i < entries.size(); i++) {
			if (entries[i] == "")
				continue;
			parse_mapping(entries[i]);
		}
	}
}

//////////////////////////////////////////////////////////
