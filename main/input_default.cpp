/*************************************************************************/
/*  input_default.cpp                                                    */
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
#include "input_default.h"

#include "input_map.h"
#include "os/os.h"
#include "scene/resources/texture.h"
#include "servers/visual_server.h"

void InputDefault::SpeedTrack::update(const Vector2 &p_delta_p) {

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

void InputDefault::SpeedTrack::reset() {
	last_tick = OS::get_singleton()->get_ticks_usec();
	speed = Vector2();
	accum_t = 0;
}

InputDefault::SpeedTrack::SpeedTrack() {

	min_ref_frame = 0.1;
	max_ref_frame = 0.3;
	reset();
}

bool InputDefault::is_key_pressed(int p_scancode) const {

	_THREAD_SAFE_METHOD_
	return keys_pressed.has(p_scancode);
}

bool InputDefault::is_mouse_button_pressed(int p_button) const {

	_THREAD_SAFE_METHOD_
	return (mouse_button_mask & (1 << (p_button - 1))) != 0;
}

static int _combine_device(int p_value, int p_device) {

	return p_value | (p_device << 20);
}

bool InputDefault::is_joy_button_pressed(int p_device, int p_button) const {

	_THREAD_SAFE_METHOD_
	return joy_buttons_pressed.has(_combine_device(p_button, p_device));
}

bool InputDefault::is_action_pressed(const StringName &p_action) const {

	return action_state.has(p_action) && action_state[p_action].pressed;
}

bool InputDefault::is_action_just_pressed(const StringName &p_action) const {

	const Map<StringName, Action>::Element *E = action_state.find(p_action);
	if (!E)
		return false;

	if (Engine::get_singleton()->is_in_physics_frame()) {
		return E->get().pressed && E->get().physics_frame == Engine::get_singleton()->get_physics_frames();
	} else {
		return E->get().pressed && E->get().idle_frame == Engine::get_singleton()->get_idle_frames();
	}
}

bool InputDefault::is_action_just_released(const StringName &p_action) const {

	const Map<StringName, Action>::Element *E = action_state.find(p_action);
	if (!E)
		return false;

	if (Engine::get_singleton()->is_in_physics_frame()) {
		return !E->get().pressed && E->get().physics_frame == Engine::get_singleton()->get_physics_frames();
	} else {
		return !E->get().pressed && E->get().idle_frame == Engine::get_singleton()->get_idle_frames();
	}
}

float InputDefault::get_joy_axis(int p_device, int p_axis) const {

	_THREAD_SAFE_METHOD_
	int c = _combine_device(p_axis, p_device);
	if (_joy_axis.has(c)) {
		return _joy_axis[c];
	} else {
		return 0;
	}
}

String InputDefault::get_joy_name(int p_idx) {

	_THREAD_SAFE_METHOD_
	return joy_names[p_idx].name;
};

Vector2 InputDefault::get_joy_vibration_strength(int p_device) {
	if (joy_vibration.has(p_device)) {
		return Vector2(joy_vibration[p_device].weak_magnitude, joy_vibration[p_device].strong_magnitude);
	} else {
		return Vector2(0, 0);
	}
}

uint64_t InputDefault::get_joy_vibration_timestamp(int p_device) {
	if (joy_vibration.has(p_device)) {
		return joy_vibration[p_device].timestamp;
	} else {
		return 0;
	}
}

float InputDefault::get_joy_vibration_duration(int p_device) {
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

void InputDefault::joy_connection_changed(int p_idx, bool p_connected, String p_name, String p_guid) {

	_THREAD_SAFE_METHOD_
	Joypad js;
	js.name = p_connected ? p_name : "";
	js.uid = p_connected ? p_guid : "";
	js.mapping = -1;
	js.hat_current = 0;

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

Vector3 InputDefault::get_gravity() const {

	_THREAD_SAFE_METHOD_
	return gravity;
}

Vector3 InputDefault::get_accelerometer() const {

	_THREAD_SAFE_METHOD_
	return accelerometer;
}

Vector3 InputDefault::get_magnetometer() const {

	_THREAD_SAFE_METHOD_
	return magnetometer;
}

Vector3 InputDefault::get_gyroscope() const {

	_THREAD_SAFE_METHOD_
	return gyroscope;
}

void InputDefault::parse_input_event(const Ref<InputEvent> &p_event) {

	_THREAD_SAFE_METHOD_

	Ref<InputEventKey> k = p_event;
	if (k.is_valid() && !k->is_echo() && k->get_scancode() != 0) {

		//print_line(p_event);

		if (k->is_pressed())
			keys_pressed.insert(k->get_scancode());
		else
			keys_pressed.erase(k->get_scancode());
	}

	Ref<InputEventMouseButton> mb = p_event;

	if (mb.is_valid() && !mb->is_doubleclick()) {

		if (mb->is_pressed()) {
			mouse_button_mask |= (1 << (mb->get_button_index() - 1));
		} else {
			mouse_button_mask &= ~(1 << (mb->get_button_index() - 1));
		}

		if (main_loop && emulate_touch && mb->get_button_index() == 1) {
			Ref<InputEventScreenTouch> touch_event;
			touch_event.instance();
			touch_event->set_pressed(mb->is_pressed());
			touch_event->set_position(mb->get_position());
			main_loop->input_event(touch_event);
		}

		Point2 pos = mb->get_global_position();
		if (mouse_pos != pos) {
			set_mouse_position(pos);
		}
	}

	Ref<InputEventMouseMotion> mm = p_event;

	if (mm.is_valid()) {

		if (main_loop && emulate_touch && mm->get_button_mask() & 1) {
			Ref<InputEventScreenDrag> drag_event;
			drag_event.instance();

			drag_event->set_position(mm->get_position());
			drag_event->set_relative(mm->get_relative());
			drag_event->set_speed(mm->get_speed());

			main_loop->input_event(drag_event);
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

	if (!p_event->is_echo()) {
		for (const Map<StringName, InputMap::Action>::Element *E = InputMap::get_singleton()->get_action_map().front(); E; E = E->next()) {

			if (InputMap::get_singleton()->event_is_action(p_event, E->key()) && is_action_pressed(E->key()) != p_event->is_pressed()) {
				Action action;
				action.physics_frame = Engine::get_singleton()->get_physics_frames();
				action.idle_frame = Engine::get_singleton()->get_idle_frames();
				action.pressed = p_event->is_pressed();
				action_state[E->key()] = action;
			}
		}
	}

	if (main_loop)
		main_loop->input_event(p_event);
}

void InputDefault::set_joy_axis(int p_device, int p_axis, float p_value) {

	_THREAD_SAFE_METHOD_
	int c = _combine_device(p_axis, p_device);
	_joy_axis[c] = p_value;
}

void InputDefault::start_joy_vibration(int p_device, float p_weak_magnitude, float p_strong_magnitude, float p_duration) {
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

void InputDefault::stop_joy_vibration(int p_device) {
	_THREAD_SAFE_METHOD_
	VibrationInfo vibration;
	vibration.weak_magnitude = 0;
	vibration.strong_magnitude = 0;
	vibration.duration = 0;
	vibration.timestamp = OS::get_singleton()->get_ticks_usec();
	joy_vibration[p_device] = vibration;
}

void InputDefault::set_gravity(const Vector3 &p_gravity) {

	_THREAD_SAFE_METHOD_

	gravity = p_gravity;
}

void InputDefault::set_accelerometer(const Vector3 &p_accel) {

	_THREAD_SAFE_METHOD_

	accelerometer = p_accel;
}

void InputDefault::set_magnetometer(const Vector3 &p_magnetometer) {

	_THREAD_SAFE_METHOD_

	magnetometer = p_magnetometer;
}

void InputDefault::set_gyroscope(const Vector3 &p_gyroscope) {

	_THREAD_SAFE_METHOD_

	gyroscope = p_gyroscope;
}

void InputDefault::set_main_loop(MainLoop *p_main_loop) {
	main_loop = p_main_loop;
}

void InputDefault::set_mouse_position(const Point2 &p_posf) {

	mouse_speed_track.update(p_posf - mouse_pos);
	mouse_pos = p_posf;
	if (custom_cursor.is_valid()) {
		//removed, please insist that we implement hardware cursors
		//		VisualServer::get_singleton()->cursor_set_pos(get_mouse_position());
	}
}

Point2 InputDefault::get_mouse_position() const {

	return mouse_pos;
}
Point2 InputDefault::get_last_mouse_speed() const {

	return mouse_speed_track.speed;
}

int InputDefault::get_mouse_button_mask() const {

	return mouse_button_mask; // do not trust OS implementaiton, should remove it - OS::get_singleton()->get_mouse_button_state();
}

void InputDefault::warp_mouse_position(const Vector2 &p_to) {

	OS::get_singleton()->warp_mouse_position(p_to);
}

Point2i InputDefault::warp_mouse_motion(const Ref<InputEventMouseMotion> &p_motion, const Rect2 &p_rect) {

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
		OS::get_singleton()->warp_mouse_position(pos_warped + p_rect.position);
	}

	return rel_warped;
}

void InputDefault::iteration(float p_step) {
}

void InputDefault::action_press(const StringName &p_action) {

	Action action;

	action.physics_frame = Engine::get_singleton()->get_physics_frames();
	action.idle_frame = Engine::get_singleton()->get_idle_frames();
	action.pressed = true;

	action_state[p_action] = action;
}

void InputDefault::action_release(const StringName &p_action) {

	Action action;

	action.physics_frame = Engine::get_singleton()->get_physics_frames();
	action.idle_frame = Engine::get_singleton()->get_idle_frames();
	action.pressed = false;

	action_state[p_action] = action;
}

void InputDefault::set_emulate_touch(bool p_emulate) {

	emulate_touch = p_emulate;
}

bool InputDefault::is_emulating_touchscreen() const {

	return emulate_touch;
}

void InputDefault::set_custom_mouse_cursor(const RES &p_cursor, const Vector2 &p_hotspot) {
	/* no longer supported, leaving this for reference to anyone who might want to implement hardware cursors
	if (custom_cursor == p_cursor)
		return;

	custom_cursor = p_cursor;

	if (p_cursor.is_null()) {
		set_mouse_mode(MOUSE_MODE_VISIBLE);
		//removed, please insist us to implement hardare cursors
		//VisualServer::get_singleton()->cursor_set_visible(false);
	} else {
		Ref<AtlasTexture> atex = custom_cursor;
		Rect2 region = atex.is_valid() ? atex->get_region() : Rect2();
		set_mouse_mode(MOUSE_MODE_HIDDEN);
		VisualServer::get_singleton()->cursor_set_visible(true);
		VisualServer::get_singleton()->cursor_set_texture(custom_cursor->get_rid(), p_hotspot, 0, region);
		VisualServer::get_singleton()->cursor_set_pos(get_mouse_position());
	}
	*/
}

void InputDefault::set_mouse_in_window(bool p_in_window) {
	/* no longer supported, leaving this for reference to anyone who might want to implement hardware cursors
	if (custom_cursor.is_valid()) {

		if (p_in_window) {
			set_mouse_mode(MOUSE_MODE_HIDDEN);
			VisualServer::get_singleton()->cursor_set_visible(true);
		} else {
			set_mouse_mode(MOUSE_MODE_VISIBLE);
			VisualServer::get_singleton()->cursor_set_visible(false);
		}
	}
	*/
}

// from github.com/gabomdq/SDL_GameControllerDB
static const char *s_ControllerMappings[] = {
#ifdef WINDOWS_ENABLED
	"00f00300000000000000504944564944,RetroUSB.com RetroPad,a:b1,b:b5,x:b0,y:b4,back:b2,start:b3,leftshoulder:b6,rightshoulder:b7,leftx:a0,lefty:a1,",
	"00f0f100000000000000504944564944,RetroUSB.com Super RetroPort,a:b1,b:b5,x:b0,y:b4,back:b2,start:b3,leftshoulder:b6,rightshoulder:b7,leftx:a0,lefty:a1,",
	"02200090000000000000504944564944,8Bitdo NES30 PRO USB,a:b0,b:b1,x:b3,y:b4,leftshoulder:b6,rightshoulder:b7,lefttrigger:b8,righttrigger:b9,back:b10,start:b11,leftstick:b13,rightstick:b14,leftx:a0,lefty:a1,rightx:a3,righty:a4,dpup:h0.1,dpright:h0.2,dpdown:h0.4,dpleft:h0.8,",
	"0d0f4900000000000000504944564944,Hatsune Miku Sho Controller,a:b1,b:b2,x:b0,y:b3,back:b8,guide:b12,start:b9,leftstick:b10,rightstick:b11,leftshoulder:b4,rightshoulder:b5,dpup:h0.1,dpdown:h0.4,dpleft:h0.8,dpright:h0.2,leftx:a0,lefty:a1,rightx:a2,righty:a3,lefttrigger:b6,righttrigger:b7,",
	"0d0f6e00000000000000504944564944,HORIPAD 4,a:b1,b:b2,y:b3,x:b0,start:b9,guide:b12,back:b8,leftstick:b10,rightstick:b11,leftshoulder:b4,rightshoulder:b5,dpup:h0.1,dpleft:h0.8,dpdown:h0.4,dpright:h0.2,leftx:a0,lefty:a1,rightx:a2,righty:a3,lefttrigger:b6,righttrigger:b7,",
	"10080100000000000000504944564944,PS1 USB,a:b2,b:b1,x:b3,y:b0,back:b8,start:b9,leftshoulder:b6,rightshoulder:b7,leftstick:b10,rightstick:b11,leftx:a0,lefty:a1,rightx:a3,righty:a2,lefttrigger:b4,righttrigger:b5,dpup:h0.1,dpleft:h0.8,dpdown:h0.4,dpright:h0.2,",
	"100801e5000000000000504944564944,NEXT Classic USB Game Controller,a:b0,b:b1,back:b8,start:b9,rightx:a2,righty:a3,leftx:a0,lefty:a1,",
	"10080300000000000000504944564944,PS2 USB,a:b2,b:b1,y:b0,x:b3,start:b9,back:b8,leftstick:b10,rightstick:b11,leftshoulder:b6,rightshoulder:b7,dpup:h0.1,dpleft:h0.8,dpdown:h0.4,dpright:h0.2,leftx:a0,lefty:a1,rightx:a4,righty:a2,lefttrigger:b4,righttrigger:b5,",
	"10280900000000000000504944564944,8Bitdo SFC30 GamePad,a:b1,b:b0,y:b3,x:b4,start:b11,back:b10,leftshoulder:b6,leftx:a0,lefty:a1,rightshoulder:b7,",
	"20380900000000000000504944564944,8Bitdo NES30 PRO Wireless,a:b0,b:b1,x:b3,y:b4,leftshoulder:b6,rightshoulder:b7,lefttrigger:b8,righttrigger:b9,back:b10,start:b11,leftstick:b13,rightstick:b14,leftx:a0,lefty:a1,rightx:a3,righty:a4,dpup:h0.1,dpright:h0.2,dpdown:h0.4,dpleft:h0.8,",
	"25090500000000000000504944564944,PS3 DualShock,a:b2,b:b1,back:b9,dpdown:h0.8,dpleft:h0.4,dpright:h0.2,dpup:h0.1,guide:,leftshoulder:b6,leftstick:b10,lefttrigger:b4,leftx:a0,lefty:a1,rightshoulder:b7,rightstick:b11,righttrigger:b5,rightx:a2,righty:a3,start:b8,x:b0,y:b3,",
	"2509e803000000000000504944564944,Mayflash Wii Classic Controller,a:b1,b:b0,x:b3,y:b2,back:b8,guide:b10,start:b9,leftshoulder:b4,rightshoulder:b5,dpup:h0.1,dpdown:h0.4,dpleft:h0.8,dpright:h0.2,dpup:b11,dpdown:b13,dpleft:b12,dpright:b14,leftx:a0,lefty:a1,rightx:a2,righty:a3,lefttrigger:b6,righttrigger:b7,",
	"28040140000000000000504944564944,GamePad Pro USB,a:b1,b:b2,x:b0,y:b3,back:b8,start:b9,leftshoulder:b4,rightshoulder:b5,leftx:a0,lefty:a1,lefttrigger:b6,righttrigger:b7,",
	"300f1001000000000000504944564944,Saitek P480 Rumble Pad,a:b2,b:b3,x:b0,y:b1,back:b8,start:b9,leftstick:b10,rightstick:b11,leftshoulder:b4,rightshoulder:b6,dpup:h0.1,dpdown:h0.4,dpleft:h0.8,dpright:h0.2,leftx:a0,lefty:a1,rightx:a3,righty:a2,lefttrigger:b5,righttrigger:b7,",
	"341a0108000000000000504944564944,EXEQ RF USB Gamepad 8206,a:b0,b:b1,x:b2,y:b3,leftshoulder:b4,rightshoulder:b5,leftstick:b8,rightstick:b7,back:b8,start:b9,dpdown:h0.4,dpleft:h0.8,dpright:h0.2,dpup:h0.1,leftx:a0,lefty:a1,rightx:a2,righty:a3,",
	"341a3608000000000000504944564944,Afterglow PS3 Controller,a:b1,b:b2,back:b8,dpdown:h0.4,dpleft:h0.8,dpright:h0.2,dpup:h0.1,guide:b12,leftshoulder:b4,leftstick:b10,lefttrigger:b6,leftx:a0,lefty:a1,rightshoulder:b5,rightstick:b11,righttrigger:b7,rightx:a2,righty:a3,start:b9,x:b0,y:b3,",
	"36280100000000000000504944564944,OUYA Controller,a:b0,b:b3,y:b2,x:b1,start:b14,guide:b15,leftstick:b6,rightstick:b7,leftshoulder:b4,rightshoulder:b5,dpup:b8,dpleft:b10,dpdown:b9,dpright:b11,leftx:a0,lefty:a1,rightx:a3,righty:a4,lefttrigger:b12,righttrigger:b13,",
	"49190204000000000000504944564944,Ipega PG-9023,a:b0,b:b1,x:b3,y:b4,back:b10,start:b11,leftstick:b13,rightstick:b14,leftshoulder:b6,rightshoulder:b7,dpup:h0.1,dpdown:h0.4,dpleft:h0.8,dpright:h0.2,leftx:a0,lefty:a1,rightx:a3,righty:a4,lefttrigger:b8,righttrigger:b9,",
	"4b12014d000000000000504944564944,NYKO AIRFLO,a:b0,b:b1,x:b2,y:b3,back:b8,guide:b10,start:b9,leftstick:a0,rightstick:a2,leftshoulder:a3,rightshoulder:b5,dpup:h0.1,dpdown:h0.0,dpleft:h0.8,dpright:h0.2,leftx:h0.6,lefty:h0.12,rightx:h0.9,righty:h0.4,lefttrigger:b6,righttrigger:b7,",
	"4c056802000000000000504944564944,PS3 Controller,a:b14,b:b13,back:b0,dpdown:b6,dpleft:b7,dpright:b5,dpup:b4,guide:b16,leftshoulder:b10,leftstick:b1,lefttrigger:b8,leftx:a0,lefty:a1,rightshoulder:b11,rightstick:b2,righttrigger:b9,rightx:a2,righty:a3,start:b3,x:b15,y:b12,",
	"4c05a00b000000000000504944564944,Sony DualShock 4 Wireless Adaptor,a:b1,b:b2,y:b3,x:b0,start:b9,guide:b12,back:b13,leftstick:b10,rightstick:b11,leftshoulder:b4,rightshoulder:b5,dpup:h0.1,dpleft:h0.8,dpdown:h0.4,dpright:h0.2,leftx:a0,lefty:a1,rightx:a2,righty:a5,lefttrigger:a3,righttrigger:a4,",
	"4c05c405000000000000504944564944,PS4 Controller,a:b1,b:b2,back:b8,dpdown:h0.4,dpleft:h0.8,dpright:h0.2,dpup:h0.1,guide:b12,leftshoulder:b4,leftstick:b10,lefttrigger:a3,leftx:a0,lefty:a1,rightshoulder:b5,rightstick:b11,righttrigger:a4,rightx:a2,righty:a5,start:b9,x:b0,y:b3,",
	"4c05cc09000000000000504944564944,Sony DualShock 4,a:b1,b:b2,y:b3,x:b0,start:b9,guide:b12,back:b13,leftstick:b10,rightstick:b11,leftshoulder:b4,rightshoulder:b5,dpup:h0.1,dpleft:h0.8,dpdown:h0.4,dpright:h0.2,leftx:a0,lefty:a1,rightx:a2,righty:a5,lefttrigger:a3,righttrigger:a4,",
	"4f0400b3000000000000504944564944,Thrustmaster Firestorm Dual Power,a:b0,b:b2,y:b3,x:b1,start:b10,guide:b8,back:b9,leftstick:b11,rightstick:b12,leftshoulder:b4,rightshoulder:b6,dpup:h0.1,dpleft:h0.8,dpdown:h0.4,dpright:h0.2,leftx:a0,lefty:a1,rightx:a2,righty:a3,lefttrigger:b5,righttrigger:b7,",
	"4f0415b3000000000000504944564944,Thrustmaster Dual Analog 3.2,x:b1,a:b0,b:b2,y:b3,back:b8,start:b9,dpleft:h0.8,dpdown:h0.4,dpright:h0.2,dpup:h0.1,leftshoulder:b4,lefttrigger:b5,rightshoulder:b6,righttrigger:b7,leftstick:b10,rightstick:b11,leftx:a0,lefty:a1,rightx:a2,righty:a3,",
	"4f0423b3000000000000504944564944,Dual Trigger 3-in-1,a:b1,b:b2,x:b0,y:b3,back:b8,start:b9,leftstick:b10,rightstick:b11,leftshoulder:b4,rightshoulder:b5,dpup:h0.1,dpdown:h0.4,dpleft:h0.8,dpright:h0.2,leftx:a0,lefty:a1,rightx:a2,righty:a5,lefttrigger:b6,righttrigger:b7,",
	"63252305000000000000504944564944,USB Vibration Joystick (BM),x:b3,a:b2,b:b1,y:b0,back:b8,start:b9,dpleft:h0.8,dpdown:h0.4,dpright:h0.2,dpup:h0.1,leftshoulder:b4,lefttrigger:b6,rightshoulder:b5,righttrigger:b7,leftstick:b10,rightstick:b11,leftx:a0,lefty:a1,rightx:a2,righty:a3,",
	"6d0416c2000000000000504944564944,Generic DirectInput Controller,a:b1,b:b2,back:b8,dpdown:h0.4,dpleft:h0.8,dpright:h0.2,dpup:h0.1,leftshoulder:b4,leftstick:b10,lefttrigger:b6,leftx:a0,lefty:a1,rightshoulder:b5,rightstick:b11,righttrigger:b7,rightx:a2,righty:a3,start:b9,x:b0,y:b3,",
	"6d0418c2000000000000504944564944,Logitech RumblePad 2 USB,x:b0,a:b1,b:b2,y:b3,back:b8,start:b9,dpleft:h0.8,dpdown:h0.4,dpright:h0.2,dpup:h0.1,leftshoulder:b4,lefttrigger:b6,rightshoulder:b5,righttrigger:b7,leftstick:b10,rightstick:b11,leftx:a0,lefty:a1,rightx:a2,righty:a3,",
	"6d0419c2000000000000504944564944,Logitech F710 Gamepad,a:b1,b:b2,back:b8,dpdown:h0.4,dpleft:h0.8,dpright:h0.2,dpup:h0.1,leftshoulder:b4,leftstick:b10,lefttrigger:b6,leftx:a0,lefty:a1,rightshoulder:b5,rightstick:b11,righttrigger:b7,rightx:a2,righty:a3,start:b9,x:b0,y:b3,",
	"6f0e1e01000000000000504944564944,Rock Candy Gamepad for PS3,a:b1,b:b2,x:b0,y:b3,back:b8,start:b9,guide:b12,leftshoulder:b4,rightshoulder:b5,leftstick:b10,rightstick:b11,leftx:a0,lefty:a1,rightx:a2,righty:a3,lefttrigger:b6,righttrigger:b7,dpup:h0.1,dpleft:h0.8,dpdown:h0.4,dpright:h0.2,",
	"79000018000000000000504944564944,Mayflash WiiU Pro Game Controller Adapter (DInput),a:b1,b:b2,x:b0,y:b3,back:b8,start:b9,leftstick:b10,rightstick:b11,leftshoulder:b4,rightshoulder:b5,dpup:h0.1,dpdown:h0.4,dpleft:h0.8,dpright:h0.2,leftx:a0,lefty:a1,rightx:a2,righty:a3,lefttrigger:b6,righttrigger:b7,",
	"79000600000000000000504944564944,G-Shark GS-GP702,a:b2,b:b1,x:b3,y:b0,back:b8,start:b9,leftstick:b10,rightstick:b11,leftshoulder:b4,rightshoulder:b5,dpup:h0.1,dpdown:h0.4,dpleft:h0.8,dpright:h0.2,leftx:a0,lefty:a1,rightx:a2,righty:a4,lefttrigger:b6,righttrigger:b7,",
	"79000600000000000000504944564944,NGS Phantom,a:b2,b:b3,y:b1,x:b0,start:b9,back:b8,leftstick:b10,rightstick:b11,leftshoulder:b4,rightshoulder:b5,dpup:h0.1,dpleft:h0.8,dpdown:h0.4,dpright:h0.2,leftx:a0,lefty:a1,rightx:a2,righty:a4,lefttrigger:b6,righttrigger:b7,",
	"79004318000000000000504944564944,Mayflash GameCube Controller Adapter,a:b1,b:b2,x:b0,y:b3,back:b0,start:b9,guide:b0,leftshoulder:b4,rightshoulder:b7,leftstick:b0,rightstick:b0,leftx:a0,lefty:a1,rightx:a5,righty:a2,lefttrigger:a3,righttrigger:a4,dpup:h0.1,dpleft:h0.8,dpdown:h0.4,dpright:h0.2,",
	"79000600000000000000504944564944,Generic Speedlink,a:b2,b:b1,y:b0,x:b3,start:b9,back:b8,leftstick:b10,rightstick:b11,leftshoulder:b4,rightshoulder:b5,dpup:h0.1,dpleft:h0.8,dpdown:h0.4,dpright:h0.2,leftx:a0,lefty:a1,rightx:a2,righty:a4,lefttrigger:b6,righttrigger:b7,",
	"83056020000000000000504944564944,iBuffalo USB 2-axis 8-button Gamepad,a:b1,b:b0,y:b2,x:b3,start:b7,back:b6,leftshoulder:b4,rightshoulder:b5,leftx:a0,lefty:a1,",
	"88880803000000000000504944564944,PS3 Controller,a:b2,b:b1,back:b8,dpdown:h0.8,dpleft:h0.4,dpright:h0.2,dpup:h0.1,guide:b12,leftshoulder:b4,leftstick:b9,lefttrigger:b6,leftx:a0,lefty:a1,rightshoulder:b5,rightstick:b10,righttrigger:b7,rightx:a3,righty:a4,start:b11,x:b0,y:b3,",
	"8f0e0300000000000000504944564944,Piranha xtreme,x:b3,a:b2,b:b1,y:b0,back:b8,start:b9,dpleft:h0.8,dpdown:h0.4,dpright:h0.2,dpup:h0.1,leftshoulder:b6,lefttrigger:b4,rightshoulder:b7,righttrigger:b5,leftstick:b10,rightstick:b11,leftx:a0,lefty:a1,rightx:a3,righty:a2,",
	"8f0e0d31000000000000504944564944,Multilaser JS071 USB,a:b1,b:b2,y:b3,x:b0,start:b9,back:b8,leftstick:b10,rightstick:b11,leftshoulder:b4,rightshoulder:b5,dpup:h0.1,dpleft:h0.8,dpdown:h0.4,dpright:h0.2,leftx:a0,lefty:a1,rightx:a2,righty:a3,lefttrigger:b6,righttrigger:b7,",
	"8f0e1200000000000000504944564944,Acme,x:b2,a:b0,b:b1,y:b3,back:b8,start:b9,dpleft:h0.8,dpdown:h0.4,dpright:h0.2,dpup:h0.1,leftshoulder:b4,lefttrigger:b5,rightshoulder:b6,righttrigger:b7,leftstick:b10,rightstick:b11,leftx:a0,lefty:a1,rightx:a3,righty:a2,",
	"9000318000000000000504944564944,Mayflash Wiimote PC Adapter,a:b2,b:h0.4,x:b0,y:b1,back:b4,start:b5,guide:b11,leftshoulder:b6,rightshoulder:b3,leftx:a0,lefty:a1,",
	"a3060cff000000000000504944564944,Saitek P2500,a:b2,b:b3,y:b1,x:b0,start:b4,guide:b10,back:b5,leftstick:b8,rightstick:b9,leftshoulder:b6,rightshoulder:b7,dpup:h0.1,dpleft:h0.8,dpdown:h0.4,dpright:h0.2,leftx:a0,lefty:a1,rightx:a2,righty:a3,",
	"c0111352000000000000504944564944,Battalife Joystick,x:b4,a:b6,b:b7,y:b5,back:b2,start:b3,leftshoulder:b0,rightshoulder:b1,leftx:a0,lefty:a1,",
	"c911f055000000000000504944564944,GAMEPAD,a:b0,b:b1,back:b8,dpdown:h0.4,dpleft:h0.8,dpright:h0.2,dpup:h0.1,leftshoulder:b4,leftstick:b10,lefttrigger:b6,leftx:a0,lefty:a1,rightshoulder:b5,rightstick:b11,righttrigger:b7,rightx:a2,righty:a3,start:b9,x:b2,y:b3,",
	"d6206dca000000000000504944564944,PowerA Pro Ex,a:b1,b:b2,x:b0,y:b3,back:b8,guide:b12,start:b9,leftstick:b10,rightstick:b11,leftshoulder:b4,rightshoulder:b5,dpup:h0.1,dpdown:h0.0,dpleft:h0.8,dpright:h0.2,leftx:a0,lefty:a1,rightx:a2,righty:a3,lefttrigger:b6,righttrigger:b7,",
	"ff113133000000000000504944564944,Gembird JPD-DualForce,a:b2,b:b3,x:b0,y:b1,start:b9,back:b8,leftshoulder:b4,rightshoulder:b5,dpup:h0.1,dpleft:h0.8,dpdown:h0.4,dpright:h0.2,leftx:a0,lefty:a1,rightx:a2,righty:a4,lefttrigger:b6,righttrigger:b7,leftstick:b10,rightstick:b11,",
	"ff113133000000000000504944564944,SVEN X-PAD,a:b2,b:b3,y:b1,x:b0,start:b5,back:b4,leftshoulder:b6,rightshoulder:b7,dpup:h0.1,dpleft:h0.8,dpdown:h0.4,dpright:h0.2,leftx:a0,lefty:a1,rightx:a2,righty:a4,lefttrigger:b8,righttrigger:b9,",
	"ffff0000000000000000504944564944,GameStop Gamepad,a:b0,b:b1,back:b8,dpdown:h0.4,dpleft:h0.8,dpright:h0.2,dpup:h0.1,guide:,leftshoulder:b4,leftstick:b10,lefttrigger:b6,leftx:a0,lefty:a1,rightshoulder:b5,rightstick:b11,righttrigger:b7,rightx:a2,righty:a3,start:b9,x:b2,y:b3,",
	"__XINPUT_DEVICE__,XInput Gamepad,a:b12,b:b13,x:b14,y:b15,start:b4,back:b5,leftstick:b6,rightstick:b7,leftshoulder:b8,rightshoulder:b9,dpup:b0,dpdown:b1,dpleft:b2,dpright:b3,leftx:a0,lefty:a1,rightx:a2,righty:a3,lefttrigger:a4,righttrigger:a5,",
#endif

#ifdef OSX_ENABLED
	"0500000047532047616d657061640000,GameStop Gamepad,a:b0,b:b1,back:b8,dpdown:h0.4,dpleft:h0.8,dpright:h0.2,dpup:h0.1,guide:,leftshoulder:b4,leftstick:b10,lefttrigger:b6,leftx:a0,lefty:a1,rightshoulder:b5,rightstick:b11,righttrigger:b7,rightx:a2,righty:a3,start:b9,x:b2,y:b3,",
	"050000005769696d6f74652028303000,Wii Remote,a:b4,b:b5,y:b9,x:b10,start:b6,guide:b8,back:b7,dpup:b2,dpleft:b0,dpdown:b3,dpright:b1,leftx:a0,lefty:a1,lefttrigger:b12,righttrigger:,leftshoulder:b11,",
	"050000005769696d6f74652028313800,Wii U Pro Controller,a:b16,b:b15,x:b18,y:b17,back:b7,guide:b8,start:b6,leftstick:b23,rightstick:b24,leftshoulder:b19,rightshoulder:b20,dpup:b11,dpdown:b12,dpleft:b13,dpright:b14,leftx:a0,lefty:a1,rightx:a2,righty:a3,lefttrigger:b21,righttrigger:b22,",
	"0d0f0000000000004d00000000000000,HORI Gem Pad 3,a:b1,b:b2,y:b3,x:b0,start:b9,guide:b12,back:b8,leftstick:b10,rightstick:b11,leftshoulder:b4,rightshoulder:b5,dpup:h0.1,dpleft:h0.8,dpdown:h0.4,dpright:h0.2,leftx:a0,lefty:a1,rightx:a2,righty:a3,lefttrigger:b6,righttrigger:b7,",
	"0d0f0000000000006600000000000000,HORIPAD FPS PLUS 4,a:b1,b:b2,y:b3,x:b0,start:b9,guide:b12,back:b8,leftstick:b10,rightstick:b11,leftshoulder:b4,rightshoulder:b5,dpup:h0.1,dpleft:h0.8,dpdown:h0.4,dpright:h0.2,leftx:a0,lefty:a1,rightx:a2,righty:a5,lefttrigger:b6,righttrigger:a4,",
	"10280000000000000900000000000000,8Bitdo SFC30 GamePad,a:b1,b:b0,x:b4,y:b3,back:b10,start:b11,leftshoulder:b6,rightshoulder:b7,leftx:a0,lefty:a1,",
	"2509000000000000e803000000000000,Mayflash Wii Classic Controller,a:b1,b:b0,x:b3,y:b2,back:b8,guide:b10,start:b9,leftshoulder:b4,rightshoulder:b5,dpup:h0.1,dpdown:h0.4,dpleft:h0.8,dpright:h0.2,dpup:b11,dpdown:b13,dpleft:b12,dpright:b14,leftx:a0,lefty:a1,rightx:a2,righty:a3,lefttrigger:b6,righttrigger:b7,",
	"351200000000000021ab000000000000,SFC30 Joystick,a:b1,b:b0,x:b4,y:b3,back:b10,start:b11,leftshoulder:b6,rightshoulder:b7,leftx:a0,lefty:a1,",
	"4c050000000000006802000000000000,PS3 Controller,a:b14,b:b13,back:b0,dpdown:b6,dpleft:b7,dpright:b5,dpup:b4,guide:b16,leftshoulder:b10,leftstick:b1,lefttrigger:b8,leftx:a0,lefty:a1,rightshoulder:b11,rightstick:b2,righttrigger:b9,rightx:a2,righty:a3,start:b3,x:b15,y:b12,",
	"4c05000000000000c405000000000000,PS4 Controller,a:b1,b:b2,back:b8,dpdown:h0.4,dpleft:h0.8,dpright:h0.2,dpup:h0.1,guide:b12,leftshoulder:b4,leftstick:b10,lefttrigger:a3,leftx:a0,lefty:a1,rightshoulder:b5,rightstick:b11,righttrigger:a4,rightx:a2,righty:a5,start:b9,x:b0,y:b3,",
	"4c05000000000000cc09000000000000,Sony DualShock 4 V2,a:b1,b:b2,y:b3,x:b0,start:b9,guide:b12,back:b13,leftstick:b10,rightstick:b11,leftshoulder:b4,rightshoulder:b5,dpup:h0.1,dpleft:h0.8,dpdown:h0.4,dpright:h0.2,leftx:a0,lefty:a1,rightx:a2,righty:a5,lefttrigger:a3,righttrigger:a4,",
	"4f0400000000000000b3000000000000,Thrustmaster Firestorm Dual Power,a:b0,b:b2,y:b3,x:b1,start:b10,guide:b8,back:b9,leftstick:b11,rightstick:,leftshoulder:b4,rightshoulder:b6,dpup:h0.1,dpleft:h0.8,dpdown:h0.4,dpright:h0.2,leftx:a0,lefty:a1,rightx:a2,righty:a3,lefttrigger:b5,righttrigger:b7,",
	"4f0400000000000015b3000000000000,Thrustmaster Dual Analog 3.2,x:b1,a:b0,b:b2,y:b3,back:b8,start:b9,dpleft:h0.8,dpdown:h0.4,dpright:h0.2,dpup:h0.1,leftshoulder:b4,lefttrigger:b5,rightshoulder:b6,righttrigger:b7,leftstick:b10,rightstick:b11,leftx:a0,lefty:a1,rightx:a2,righty:a3,",
	"5e040000000000008e02000000000000,X360 Controller,a:b0,b:b1,back:b9,dpdown:b12,dpleft:b13,dpright:b14,dpup:b11,guide:b10,leftshoulder:b4,leftstick:b6,lefttrigger:a2,leftx:a0,lefty:a1,rightshoulder:b5,rightstick:b7,righttrigger:a5,rightx:a3,righty:a4,start:b8,x:b2,y:b3,",
	"5e04000000000000dd02000000000000,Xbox One Wired Controller,x:b2,a:b0,b:b1,y:b3,back:b9,guide:b10,start:b8,dpleft:b13,dpdown:b12,dpright:b14,dpup:b11,leftshoulder:b4,lefttrigger:a2,rightshoulder:b5,righttrigger:a5,leftstick:b6,rightstick:b7,leftx:a0,lefty:a1,rightx:a3,righty:a4,",
	"5e04000000000000e002000000000000,Xbox Wireless Controller,x:b2,a:b0,b:b1,y:b3,back:b6,guide:b10,start:b7,dpleft:h0.8,dpdown:h0.4,dpright:h0.2,dpup:h0.1,leftshoulder:b4,lefttrigger:a2,rightshoulder:b5,righttrigger:a5,leftstick:b8,rightstick:b9,leftx:a0,lefty:a1,rightx:a3,righty:a4,",
	"5e04000000000000ea02000000000000,Xbox Wireless Controller,x:b2,a:b0,b:b1,y:b3,back:b9,guide:b10,start:b8,dpleft:b13,dpdown:b12,dpright:b14,dpup:b11,leftshoulder:b4,lefttrigger:a2,rightshoulder:b5,righttrigger:a5,leftstick:b6,rightstick:b7,leftx:a0,lefty:a1,rightx:a3,righty:a4,",
	"6d0400000000000016c2000000000000,Logitech F310 Gamepad (DInput),a:b1,b:b2,back:b8,dpdown:h0.4,dpleft:h0.8,dpright:h0.2,dpup:h0.1,leftshoulder:b4,leftstick:b10,lefttrigger:b6,leftx:a0,lefty:a1,rightshoulder:b5,rightstick:b11,righttrigger:b7,rightx:a2,righty:a3,start:b9,x:b0,y:b3,", /* Guide button doesn't seem to be sent in DInput mode. */
	"6d0400000000000018c2000000000000,Logitech F510 Gamepad (DInput),a:b1,b:b2,back:b8,dpdown:h0.4,dpleft:h0.8,dpright:h0.2,dpup:h0.1,leftshoulder:b4,leftstick:b10,lefttrigger:b6,leftx:a0,lefty:a1,rightshoulder:b5,rightstick:b11,righttrigger:b7,rightx:a2,righty:a3,start:b9,x:b0,y:b3,",
	"6d0400000000000019c2000000000000,Logitech Wireless Gamepad (DInput),a:b1,b:b2,back:b8,dpdown:h0.4,dpleft:h0.8,dpright:h0.2,dpup:h0.1,leftshoulder:b4,leftstick:b10,lefttrigger:b6,leftx:a0,lefty:a1,rightshoulder:b5,rightstick:b11,righttrigger:b7,rightx:a2,righty:a3,start:b9,x:b0,y:b3,", /* This includes F710 in DInput mode and the "Logitech Cordless RumblePad 2", at the very least. */
	"6d040000000000001fc2000000000000,Logitech F710 Gamepad (XInput),a:b0,b:b1,back:b9,dpdown:b12,dpleft:b13,dpright:b14,dpup:b11,guide:b10,leftshoulder:b4,leftstick:b6,lefttrigger:a2,leftx:a0,lefty:a1,rightshoulder:b5,rightstick:b7,righttrigger:a5,rightx:a3,righty:a4,start:b8,x:b2,y:b3,",
	"79000000000000000018000000000000,Mayflash WiiU Pro Game Controller Adapter (DInput),a:b4,b:b8,x:b0,y:b12,back:b32,start:b36,leftstick:b40,rightstick:b44,leftshoulder:b16,rightshoulder:b20,dpup:h0.1,dpdown:h0.4,dpleft:h0.8,dpright:h0.2,leftx:a0,lefty:a4,rightx:a8,righty:a12,lefttrigger:b24,righttrigger:b28,",
	"79000000000000000600000000000000,G-Shark GP-702,a:b2,b:b1,x:b3,y:b0,back:b8,start:b9,leftstick:b10,rightstick:b11,leftshoulder:b4,rightshoulder:b5,dpup:h0.1,dpdown:h0.4,dpleft:h0.8,dpright:h0.2,leftx:a0,lefty:a1,rightx:a3,righty:a4,lefttrigger:b6,righttrigger:b7,",
	"79000000000000001100000000000000,Retrolink Classic Controller,x:b3,a:b2,b:b1,y:b0,back:b8,start:b9,leftshoulder:b4,rightshoulder:b5,leftx:a3,lefty:a4,",
	"81170000000000007e05000000000000,Sega Saturn,x:b0,a:b2,b:b4,y:b6,start:b13,dpleft:b15,dpdown:b16,dpright:b14,dpup:b17,leftshoulder:b8,lefttrigger:a5,lefttrigger:b10,rightshoulder:b9,righttrigger:a4,righttrigger:b11,leftx:a0,lefty:a2,",
	"83050000000000006020000000000000,iBuffalo USB 2-axis 8-button Gamepad,a:b1,b:b0,x:b3,y:b2,back:b6,start:b7,leftshoulder:b4,rightshoulder:b5,leftx:a0,lefty:a1,",
	"891600000000000000fd000000000000,Razer Onza Tournament,a:b0,b:b1,y:b3,x:b2,start:b8,guide:b10,back:b9,leftstick:b6,rightstick:b7,leftshoulder:b4,rightshoulder:b5,dpup:b11,dpleft:b13,dpdown:b12,dpright:b14,leftx:a0,lefty:a1,rightx:a3,righty:a4,lefttrigger:a2,righttrigger:a5,",
	"8f0e0000000000000300000000000000,Piranha xtreme,x:b3,a:b2,b:b1,y:b0,back:b8,start:b9,dpleft:h0.8,dpdown:h0.4,dpright:h0.2,dpup:h0.1,leftshoulder:b6,lefttrigger:b4,rightshoulder:b7,righttrigger:b5,leftstick:b10,rightstick:b11,leftx:a0,lefty:a1,rightx:a3,righty:a2,",
	"ad1b00000000000001f9000000000000,Gamestop BB-070 X360 Controller,a:b0,b:b1,back:b9,dpdown:b12,dpleft:b13,dpright:b14,dpup:b11,guide:b10,leftshoulder:b4,leftstick:b6,lefttrigger:a2,leftx:a0,lefty:a1,rightshoulder:b5,rightstick:b7,righttrigger:a5,rightx:a3,righty:a4,start:b8,x:b2,y:b3,",
	"b4040000000000000a01000000000000,Sega Saturn USB Gamepad,a:b0,b:b1,x:b3,y:b4,back:b5,guide:b2,start:b8,leftshoulder:b6,rightshoulder:b7,leftx:a0,lefty:a1,",
	"bd1200000000000015d0000000000000,Tomee SNES USB Controller,x:b3,a:b2,b:b1,y:b0,back:b8,start:b9,leftshoulder:b4,rightshoulder:b5,leftx:a0,lefty:a1,",
	"d814000000000000cecf000000000000,MC Cthulhu,leftx:,lefty:,rightx:,righty:,lefttrigger:b6,a:b1,b:b2,y:b3,x:b0,start:b9,back:b8,leftshoulder:b4,rightshoulder:b5,dpup:h0.1,dpleft:h0.8,dpdown:h0.4,dpright:h0.2,righttrigger:b7,",
#endif

#if X11_ENABLED
	"0000000058626f782047616d65706100,Xbox Gamepad (userspace driver),a:b0,b:b1,x:b2,y:b3,start:b7,back:b6,guide:b8,dpup:h0.1,dpdown:h0.4,dpleft:h0.8,dpright:h0.2,leftshoulder:b4,rightshoulder:b5,lefttrigger:a5,righttrigger:a4,leftstick:b9,rightstick:b10,leftx:a0,lefty:a1,rightx:a2,righty:a3,",
	"0300000000f000000300000000010000,RetroUSB.com RetroPad,a:b1,b:b5,x:b0,y:b4,back:b2,start:b3,leftshoulder:b6,rightshoulder:b7,leftx:a0,lefty:a1,",
	"0300000000f00000f100000000010000,RetroUSB.com Super RetroPort,a:b1,b:b5,x:b0,y:b4,back:b2,start:b3,leftshoulder:b6,rightshoulder:b7,leftx:a0,lefty:a1,",
	"030000000d0f00000d00000000010000,hori,a:b0,b:b6,y:b2,x:b1,dpup:h0.1,dpleft:h0.8,dpdown:h0.4,dpright:h0.2,start:b9,guide:b10,back:b8,leftshoulder:b3,rightshoulder:b7,leftx:b4,lefty:b5,",
	"030000000d0f00001000000011010000,HORI CO.,LTD. FIGHTING STICK 3,x:b0,a:b1,b:b2,y:b3,back:b8,guide:b12,start:b9,dpleft:h0.8,dpdown:h0.4,dpright:h0.2,dpup:h0.1,leftshoulder:b4,lefttrigger:b6,rightshoulder:b5,righttrigger:b7",
	"030000000d0f00002200000011010000,HORI CO.,LTD. REAL ARCADE Pro.V3,x:b0,a:b1,b:b2,y:b3,back:b8,guide:b12,start:b9,leftshoulder:b4,lefttrigger:b6,rightshoulder:b5,righttrigger:b7,dpleft:h0.8,dpdown:h0.4,dpright:h0.2,dpup:h0.1,",
	"030000000d0f00004d00000011010000,HORI Gem Pad 3,x:b0,a:b1,b:b2,y:b3,back:b8,guide:b12,start:b9,dpleft:h0.8,dpdown:h0.4,dpright:h0.2,dpup:h0.1,leftshoulder:b4,lefttrigger:b6,rightshoulder:b5,righttrigger:b7,leftstick:b10,rightstick:b11,leftx:a0,lefty:a1,rightx:a2,righty:a3,",
	"03000000100800000100000010010000,Twin USB PS2 Adapter,a:b2,b:b1,y:b0,x:b3,start:b9,guide:,back:b8,leftstick:b10,rightstick:b11,leftshoulder:b6,rightshoulder:b7,dpup:h0.1,dpleft:h0.8,dpdown:h0.4,dpright:h0.2,leftx:a0,lefty:a1,rightx:a3,righty:a2,lefttrigger:b4,righttrigger:b5,",
	"030000001008000001e5000010010000,NEXT Classic USB Game Controller,a:b0,b:b1,back:b8,start:b9,rightx:a2,righty:a3,leftx:a0,lefty:a1,",
	"03000000100800000300000010010000,USB Gamepad,a:b2,b:b1,x:b3,y:b0,start:b9,back:b8,leftstick:b10,rightstick:b11,leftshoulder:b6,rightshoulder:b7,dpup:h0.1,dpleft:h0.8,dpdown:h0.4,dpright:h0.2,leftx:a0,lefty:a1,rightx:a3,righty:a2,lefttrigger:b4,righttrigger:b5,",
	"03000000250900000500000000010000,Sony PS2 pad with SmartJoy adapter,a:b2,b:b1,y:b0,x:b3,start:b8,back:b9,leftstick:b10,rightstick:b11,leftshoulder:b6,rightshoulder:b7,dpup:h0.1,dpleft:h0.8,dpdown:h0.4,dpright:h0.2,leftx:a0,lefty:a1,rightx:a2,righty:a3,lefttrigger:b4,righttrigger:b5,",
	"03000000260900008888000000010000,GameCube {WiseGroup USB box},a:b0,b:b2,y:b3,x:b1,start:b7,leftshoulder:,rightshoulder:b6,dpup:h0.1,dpleft:h0.8,rightstick:,dpdown:h0.4,dpright:h0.2,leftx:a0,lefty:a1,rightx:a2,righty:a3,lefttrigger:a4,righttrigger:a5,",
	"03000000280400000140000000010000,Gravis GamePad Pro USB ,x:b0,a:b1,b:b2,y:b3,back:b8,start:b9,leftshoulder:b4,lefttrigger:b6,rightshoulder:b5,righttrigger:b7,leftx:a0,lefty:a1,",
	"03000000341a000005f7000010010000,GameCube {HuiJia USB box},a:b1,b:b2,y:b3,x:b0,start:b9,guide:,back:,leftstick:,rightstick:,leftshoulder:,dpleft:b15,dpdown:b14,dpright:b13,leftx:a0,lefty:a1,rightx:a5,righty:a2,lefttrigger:a3,righttrigger:a4,rightshoulder:b7,dpup:b12,",
	"03000000380700001647000010040000,Mad Catz Wired Xbox 360 Controller,x:b2,a:b0,b:b1,y:b3,back:b6,guide:b8,start:b7,dpleft:h0.8,dpdown:h0.0,dpdown:h0.4,dpright:h0.0,dpright:h0.2,dpup:h0.0,dpup:h0.1,leftshoulder:h0.0,leftshoulder:b4,lefttrigger:a2,rightshoulder:b5,righttrigger:a5,leftstick:b9,rightstick:b10,leftx:a0,lefty:a1,rightx:a3,righty:a4,",
	"030000004c0500006802000011010000,PS3 Controller,a:b14,b:b13,back:b0,dpdown:b6,dpleft:b7,dpright:b5,dpup:b4,guide:b16,leftshoulder:b10,leftstick:b1,lefttrigger:b8,leftx:a0,lefty:a1,rightshoulder:b11,rightstick:b2,righttrigger:b9,rightx:a2,righty:a3,start:b3,x:b15,y:b12,",
	"030000004c050000a00b000011010000,Sony DualShock 4 Wireless Adaptor,a:b1,b:b2,y:b3,x:b0,start:b9,guide:b12,back:b13,leftstick:b10,rightstick:b11,leftshoulder:b4,rightshoulder:b5,dpup:h0.1,dpleft:h0.8,dpdown:h0.4,dpright:h0.2,leftx:a0,lefty:a1,rightx:a2,righty:a5,lefttrigger:a3,righttrigger:a4,",
	"030000004c050000c405000011010000,Sony DualShock 4,a:b1,b:b2,y:b3,x:b0,start:b9,guide:b12,back:b8,leftstick:b10,rightstick:b11,leftshoulder:b4,rightshoulder:b5,dpup:h0.1,dpleft:h0.8,dpdown:h0.4,dpright:h0.2,leftx:a0,lefty:a1,rightx:a2,righty:a5,lefttrigger:a3,righttrigger:a4,",
	"030000004c050000c405000011810000,Sony Computer Entertainment Wireless Controller,leftx:a0,lefty:a1,dpdown:h0.4,rightstick:h0.1,rightshoulder:b5,rightx:a3,start:b9,righty:a4,dpleft:h0.8,lefttrigger:a2,x:b3,dpup:h0.1,back:b8,leftstick:b11,leftshoulder:b4,y:b2,a:b0,dpright:h0.2,righttrigger:a5,b:b1,",
	"030000004c050000cc09000011010000,Sony DualShock 4 V2,a:b1,b:b2,y:b3,x:b0,start:b9,guide:b12,back:b13,leftstick:b10,rightstick:b11,leftshoulder:b4,rightshoulder:b5,dpup:h0.1,dpleft:h0.8,dpdown:h0.4,dpright:h0.2,leftx:a0,lefty:a1,rightx:a2,righty:a5,lefttrigger:a3,righttrigger:a4,",
	"030000004f04000000b3000010010000,Thrustmaster Firestorm Dual Power,a:b0,b:b2,y:b3,x:b1,start:b10,guide:b8,back:b9,leftstick:b11,rightstick:b12,leftshoulder:b4,rightshoulder:b6,dpup:h0.1,dpleft:h0.8,dpdown:h0.4,dpright:h0.2,leftx:a0,lefty:a1,rightx:a2,righty:a3,lefttrigger:b5,righttrigger:b7,",
	"030000004f04000008d0000000010000,Thrustmaster Run N Drive  Wireless,a:b1,b:b2,x:b0,y:b3,start:b9,back:b8,leftstick:b10,rightstick:b11,leftshoulder:b4,rightshoulder:b5,dpup:h0.1,dpleft:h0.8,dpdown:h0.4,dpright:h0.2,leftx:a0,lefty:a1,rightx:a2,righty:a5,lefttrigger:b6,righttrigger:b7,",
	"030000004f04000009d0000000010000,Thrustmaster Run N Drive Wireless PS3,a:b1,b:b2,x:b0,y:b3,start:b9,guide:b12,back:b8,leftstick:b10,rightstick:b11,leftshoulder:b4,rightshoulder:b5,dpup:h0.1,dpleft:h0.8,dpdown:h0.4,dpright:h0.2,leftx:a0,lefty:a1,rightx:a2,righty:a3,lefttrigger:b6,righttrigger:b7,",
	"030000004f04000015b3000010010000,Thrustmaster Dual Analog 4,a:b0,b:b2,x:b1,y:b3,start:b9,back:b8,leftstick:b10,rightstick:b11,leftshoulder:b4,rightshoulder:b6,dpup:h0.1,dpleft:h0.8,dpdown:h0.4,dpright:h0.2,leftx:a0,lefty:a1,rightx:a2,righty:a3,lefttrigger:b5,righttrigger:b7,",
	"030000004f04000020b3000010010000,Thrustmaster 2 in 1 DT,a:b0,b:b2,y:b3,x:b1,start:b9,guide:,back:b8,leftstick:b10,rightstick:b11,leftshoulder:b4,rightshoulder:b6,dpup:h0.1,dpleft:h0.8,dpdown:h0.4,dpright:h0.2,leftx:a0,lefty:a1,rightx:a2,righty:a3,lefttrigger:b5,righttrigger:b7,",
	"030000004f04000023b3000000010000,Thrustmaster Dual Trigger 3-in-1,x:b0,a:b1,b:b2,y:b3,back:b8,start:b9,dpleft:h0.8,dpdown:h0.0,dpdown:h0.4,dpright:h0.0,dpright:h0.2,dpup:h0.0,dpup:h0.1,leftshoulder:h0.0,leftshoulder:b4,lefttrigger:b6,rightshoulder:b5,righttrigger:b7,leftstick:b10,rightstick:b11,leftx:a0,lefty:a1,rightx:a2,righty:a5,",
	"030000005e0400001907000000010000,X360 Wireless Controller,a:b0,b:b1,back:b6,dpdown:b14,dpleft:b11,dpright:b12,dpup:b13,guide:b8,leftshoulder:b4,leftstick:b9,lefttrigger:a2,leftx:a0,lefty:a1,rightshoulder:b5,rightstick:b10,righttrigger:a5,rightx:a3,righty:a4,start:b7,x:b2,y:b3,",
	"030000005e0400008502000000010000,Microsoft X-Box pad (Japan),x:b3,a:b0,b:b1,y:b4,back:b6,start:b7,dpleft:h0.8,dpdown:h0.4,dpright:h0.2,dpup:h0.1,leftshoulder:b5,lefttrigger:a2,rightshoulder:b2,righttrigger:a5,leftstick:b8,rightstick:b9,leftx:a0,lefty:a1,rightx:a3,righty:a4,",
	"030000005e0400008902000021010000,Microsoft X-Box pad v2 (US),x:b3,a:b0,b:b1,y:b4,back:b6,start:b7,dpleft:h0.8,dpdown:h0.4,dpright:h0.2,dpup:h0.1,leftshoulder:b5,lefttrigger:a2,rightshoulder:b2,righttrigger:a5,leftstick:b8,rightstick:b9,leftx:a0,lefty:a1,rightx:a3,righty:a4,",
	"030000005e0400008e02000001000000,Microsoft X-Box 360 pad,leftstick:b9,leftx:a0,lefty:a1,dpdown:h0.1,rightstick:b10,rightshoulder:b5,rightx:a3,start:b7,righty:a4,dpleft:h0.2,lefttrigger:a2,x:b2,dpup:h0.4,back:b6,leftshoulder:b4,y:b3,a:b0,dpright:h0.8,righttrigger:a5,b:b1,",
	"030000005e0400008e02000004010000,Microsoft X-Box 360 pad,a:b0,b:b1,x:b2,y:b3,back:b6,start:b7,guide:b8,leftshoulder:b4,rightshoulder:b5,leftstick:b9,rightstick:b10,leftx:a0,lefty:a1,rightx:a3,righty:a4,lefttrigger:a2,righttrigger:a5,dpup:h0.1,dpleft:h0.8,dpdown:h0.4,dpright:h0.2,",
	"030000005e0400008e02000010010000,X360 Controller,a:b0,b:b1,back:b6,dpdown:h0.4,dpleft:h0.8,dpright:h0.2,dpup:h0.1,guide:b8,leftshoulder:b4,leftstick:b9,lefttrigger:a2,leftx:a0,lefty:a1,rightshoulder:b5,rightstick:b10,righttrigger:a5,rightx:a3,righty:a4,start:b7,x:b2,y:b3,",
	"030000005e0400008e02000014010000,X360 Controller,a:b0,b:b1,back:b6,dpdown:h0.4,dpleft:h0.8,dpright:h0.2,dpup:h0.1,guide:b8,leftshoulder:b4,leftstick:b9,lefttrigger:a2,leftx:a0,lefty:a1,rightshoulder:b5,rightstick:b10,righttrigger:a5,rightx:a3,righty:a4,start:b7,x:b2,y:b3,",
	"030000005e0400008e02000020200000,SpeedLink XEOX Pro Analog Gamepad pad,x:b2,a:b0,b:b1,y:b3,back:b6,guide:b8,start:b7,dpleft:h0.8,dpdown:h0.4,dpright:h0.2,dpup:h0.1,leftshoulder:b4,lefttrigger:a2,rightshoulder:b5,righttrigger:a5,leftstick:b9,rightstick:b10,leftx:a0,lefty:a1,rightx:a3,righty:a4,",
	"030000005e0400008e02000062230000,Microsoft X-Box 360 pad,x:b2,a:b0,b:b1,y:b3,back:b6,guide:b8,start:b7,dpleft:h0.8,dpdown:h0.4,dpright:h0.2,dpup:h0.1,leftshoulder:b4,lefttrigger:a2,rightshoulder:b5,righttrigger:a5,leftstick:b9,rightstick:b10,leftx:a0,lefty:a1,rightx:a3,righty:a4,",
	"030000005e0400008e02000073050000,Speedlink TORID Wireless Gamepad,x:b2,a:b0,b:b1,y:b3,back:b6,guide:b8,start:b7,dpleft:h0.8,dpdown:h0.4,dpright:h0.2,dpup:h0.1,leftshoulder:b4,lefttrigger:a2,rightshoulder:b5,righttrigger:a5,leftstick:b9,rightstick:b10,leftx:a0,lefty:a1,rightx:a3,righty:a4,",
	"030000005e0400009102000007010000,X360 Wireless Controller,a:b0,b:b1,back:b6,dpdown:b14,dpleft:b11,dpright:b12,dpup:b13,guide:b8,leftshoulder:b4,leftstick:b9,lefttrigger:a2,leftx:a0,lefty:a1,rightshoulder:b5,rightstick:b10,righttrigger:a5,rightx:a3,righty:a4,start:b7,x:b2,y:b3,",
	"030000005e040000a102000000010000,X360 Wireless Controller,a:b0,b:b1,back:b6,dpdown:b14,dpleft:b11,dpright:b12,dpup:b13,guide:b8,leftshoulder:b4,leftstick:b9,lefttrigger:a2,leftx:a0,lefty:a1,rightshoulder:b5,rightstick:b10,righttrigger:a5,rightx:a3,righty:a4,start:b7,x:b2,y:b3,",
	"030000005e040000d102000001010000,Microsoft X-Box One pad,x:b2,a:b0,b:b1,y:b3,back:b6,guide:b8,start:b7,dpleft:h0.8,dpdown:h0.0,dpdown:h0.4,dpright:h0.0,dpright:h0.2,dpup:h0.0,dpup:h0.1,leftshoulder:h0.0,leftshoulder:b4,lefttrigger:a2,rightshoulder:b5,righttrigger:a5,leftstick:b9,rightstick:b10,leftx:a0,lefty:a1,rightx:a3,righty:a4,",
	"030000005e040000dd02000003020000,Microsoft X-Box One pad v2,x:b2,a:b0,b:b1,y:b3,back:b6,guide:b8,start:b7,dpleft:h0.8,dpdown:h0.0,dpdown:h0.4,dpright:h0.0,dpright:h0.2,dpup:h0.0,dpup:h0.1,leftshoulder:h0.0,leftshoulder:b4,lefttrigger:a2,rightshoulder:b5,righttrigger:a5,leftstick:b9,rightstick:b10,leftx:a0,lefty:a1,rightx:a3,righty:a4,",
	"03000000666600000488000000010000,Super Joy Box 5 Pro,a:b2,b:b1,x:b3,y:b0,back:b9,start:b8,leftshoulder:b6,rightshoulder:b7,leftstick:b10,rightstick:b11,leftx:a0,lefty:a1,rightx:a2,righty:a3,lefttrigger:b4,righttrigger:b5,dpup:b12,dpleft:b15,dpdown:b14,dpright:b13,",
	"030000006d04000011c2000010010000,Logitech WingMan Cordless RumblePad,a:b0,b:b1,y:b4,x:b3,start:b8,guide:b5,back:b2,leftshoulder:b6,rightshoulder:b7,dpup:h0.1,dpleft:h0.8,dpdown:h0.4,dpright:h0.2,leftx:a0,lefty:a1,rightx:a3,righty:a4,lefttrigger:b9,righttrigger:b10,",
	"030000006d04000016c2000010010000,Logitech Logitech Dual Action,x:b0,a:b1,b:b2,y:b3,back:b8,start:b9,dpleft:h0.8,dpdown:h0.0,dpdown:h0.4,dpright:h0.0,dpright:h0.2,dpup:h0.0,dpup:h0.1,leftshoulder:h0.0,dpup:h0.1,leftshoulder:h0.0,leftshoulder:b4,lefttrigger:b6,rightshoulder:b5,righttrigger:b7,leftstick:b10,rightstick:b11,leftx:a0,lefty:a1,rightx:a2,righty:a3,",
	"030000006d04000016c2000011010000,Logitech F310 Gamepad (DInput),x:b0,a:b1,b:b2,y:b3,back:b8,start:b9,dpleft:h0.8,dpdown:h0.0,dpdown:h0.4,dpright:h0.0,dpright:h0.2,dpup:h0.0,dpup:h0.1,leftshoulder:h0.0,dpup:h0.1,leftshoulder:h0.0,leftshoulder:b4,lefttrigger:b6,rightshoulder:b5,righttrigger:b7,leftstick:b10,rightstick:b11,leftx:a0,lefty:a1,rightx:a2,righty:a3,",
	"030000006d04000018c2000010010000,Logitech Logitech RumblePad 2 USB,x:b0,a:b1,b:b2,y:b3,back:b8,start:b9,dpleft:h0.8,dpdown:h0.4,dpright:h0.2,dpup:h0.1,leftshoulder:b4,lefttrigger:b6,rightshoulder:b5,righttrigger:b7,leftstick:b10,rightstick:b11,leftx:a0,lefty:a1,rightx:a2,righty:a3,",
	"030000006d04000019c2000010010000,Logitech Cordless RumblePad 2,a:b1,b:b2,back:b8,dpdown:h0.4,dpleft:h0.8,dpright:h0.2,dpup:h0.1,guide:,leftshoulder:b4,leftstick:b10,lefttrigger:b6,leftx:a0,lefty:a1,rightshoulder:b5,rightstick:b11,righttrigger:b7,rightx:a2,righty:a3,start:b9,x:b0,y:b3,",
	"030000006d04000019c2000011010000,Logitech F710 Gamepad (DInput),a:b1,b:b2,back:b8,dpdown:h0.4,dpleft:h0.8,dpright:h0.2,dpup:h0.1,leftshoulder:b4,leftstick:b10,lefttrigger:b6,leftx:a0,lefty:a1,rightshoulder:b5,rightstick:b11,righttrigger:b7,rightx:a2,righty:a3,start:b9,x:b0,y:b3,",
	"030000006d0400001dc2000014400000,Logitech F310 Gamepad (XInput),a:b0,b:b1,back:b6,dpdown:h0.4,dpleft:h0.8,dpright:h0.2,dpup:h0.1,guide:b8,leftshoulder:b4,leftstick:b9,lefttrigger:a2,leftx:a0,lefty:a1,rightshoulder:b5,rightstick:b10,righttrigger:a5,rightx:a3,righty:a4,start:b7,x:b2,y:b3,",
	"030000006d0400001ec2000020200000,Logitech F510 Gamepad (XInput),a:b0,b:b1,back:b6,dpdown:h0.4,dpleft:h0.8,dpright:h0.2,dpup:h0.1,guide:b8,leftshoulder:b4,leftstick:b9,lefttrigger:a2,leftx:a0,lefty:a1,rightshoulder:b5,rightstick:b10,righttrigger:a5,rightx:a3,righty:a4,start:b7,x:b2,y:b3,",
	"030000006d0400001fc2000005030000,Logitech F710 Gamepad (XInput),a:b0,b:b1,back:b6,dpdown:h0.4,dpleft:h0.8,dpright:h0.2,dpup:h0.1,guide:b8,leftshoulder:b4,leftstick:b9,lefttrigger:a2,leftx:a0,lefty:a1,rightshoulder:b5,rightstick:b10,righttrigger:a5,rightx:a3,righty:a4,start:b7,x:b2,y:b3,",
	"030000006e0500000320000010010000,JC-U3613M - DirectInput Mode,x:b0,a:b2,b:b3,y:b1,back:b10,guide:b12,start:b11,dpleft:h0.8,dpdown:h0.4,dpright:h0.2,dpup:h0.1,leftshoulder:b4,lefttrigger:b6,rightshoulder:b5,righttrigger:b7,leftstick:b8,rightstick:b9,leftx:a0,lefty:a1,rightx:a2,righty:a3,",
	"030000006f0e00000103000000020000,Logic3 Controller,x:b2,a:b0,b:b1,y:b3,back:b6,guide:b8,start:b7,dpleft:h0.8,dpdown:h0.0,dpdown:h0.4,dpright:h0.0,dpright:h0.2,dpup:h0.0,dpup:h0.1,leftshoulder:h0.0,leftshoulder:b4,lefttrigger:a2,rightshoulder:b5,righttrigger:a5,leftstick:b9,rightstick:b10,leftx:a0,lefty:a1,rightx:a3,righty:a4,",
	"030000006f0e00001304000000010000,Generic X-Box pad,x:b2,a:b0,b:b1,y:b3,back:b6,guide:b8,start:b7,dpleft:h0.8,dpdown:h0.4,dpright:h0.2,dpup:h0.1,leftshoulder:b4,lefttrigger:a2,rightshoulder:b5,righttrigger:a5,leftstick:b9,rightstick:b10,leftx:a0,lefty:a1,rightx:a3,righty:a4,",
	"030000006f0e00001e01000011010000,Rock Candy Gamepad for PS3,a:b1,b:b2,x:b0,y:b3,back:b8,start:b9,guide:b12,leftshoulder:b4,rightshoulder:b5,leftstick:b10,rightstick:b11,leftx:a0,lefty:a1,rightx:a2,righty:a3,lefttrigger:b6,righttrigger:b7,dpup:h0.1,dpleft:h0.8,dpdown:h0.4,dpright:h0.2,",
	"030000006f0e00001f01000000010000,Generic X-Box pad,x:b2,a:b0,b:b1,y:b3,back:b6,guide:b8,start:b7,dpleft:h0.8,dpdown:h0.4,dpright:h0.2,dpup:h0.1,leftshoulder:b4,lefttrigger:a2,rightshoulder:b5,righttrigger:a5,leftstick:b9,rightstick:b10,leftx:a0,lefty:a1,rightx:a3,righty:a4,",
	"030000006f0e00002801000011010000,PDP Rock Candy Wireless Controller for PS3,leftx:a0,lefty:a1,dpdown:h0.4,rightstick:b11,rightshoulder:b5,rightx:a2,start:b9,righty:a3,dpleft:h0.8,lefttrigger:b6,x:b0,dpup:h0.1,back:b8,leftstick:b10,leftshoulder:b4,y:b3,a:b1,dpright:h0.2,righttrigger:b7,b:b2,",
	"030000006f0e00003001000001010000,EA Sports PS3 Controller,a:b1,b:b2,y:b3,x:b0,start:b9,guide:b12,back:b8,leftstick:b10,rightstick:b11,leftshoulder:b4,rightshoulder:b5,dpup:h0.1,dpleft:h0.8,dpdown:h0.4,dpright:h0.2,leftx:a0,lefty:a1,rightx:a2,righty:a3,lefttrigger:b6,righttrigger:b7,",
	"030000006f0e00003901000020060000,Afterglow Wired Controller for Xbox One,x:b2,a:b0,b:b1,y:b3,back:b6,guide:b8,start:b7,dpleft:h0.8,dpdown:h0.4,dpright:h0.2,dpup:h0.1,leftshoulder:b4,lefttrigger:a2,rightshoulder:b5,righttrigger:a5,leftstick:b9,rightstick:b10,leftx:a0,lefty:a1,rightx:a3,righty:a4,",
	"030000006f0e00004601000001010000,Rock Candy Wired Controller for Xbox One,a:b0,b:b1,x:b2,y:b3,leftshoulder:b4,rightshoulder:b5,back:b6,start:b7,guide:b8,leftstick:b9,rightstick:b10,lefttrigger:a2,righttrigger:a5,leftx:a0,lefty:a1,rightx:a3,righty:a4,",
	"03000000790000000600000010010000,DragonRise Inc. Generic USB Joystick,x:b3,a:b2,b:b1,y:b0,back:b8,start:b9,dpleft:h0.8,dpdown:h0.4,dpright:h0.2,dpup:h0.1,leftshoulder:b4,lefttrigger:b6,rightshoulder:b5,righttrigger:b7,leftstick:b10,rightstick:b11,leftx:a0,lefty:a1,rightx:a3,righty:a4,",
	"03000000790000001100000010010000,Retrolink Classic Controller,x:b3,a:b2,b:b1,y:b0,back:b8,start:b9,leftshoulder:b4,rightshoulder:b5,leftx:a0,lefty:a1,",
	"03000000790000001100000010010000,RetroLink Saturn Classic Controller,x:b3,a:b0,b:b1,y:b4,back:b5,guide:b2,start:b8,leftshoulder:b6,rightshoulder:b7,leftx:a0,lefty:a1,",
	"03000000830500006020000010010000,iBuffalo USB 2-axis 8-button Gamepad,a:b1,b:b0,x:b3,y:b2,back:b6,start:b7,leftshoulder:b4,rightshoulder:b5,leftx:a0,lefty:a1,",
	"030000008916000000fd000024010000,Razer Onza Tournament,a:b0,b:b1,y:b3,x:b2,start:b7,guide:b8,back:b6,leftstick:b9,rightstick:b10,leftshoulder:b4,rightshoulder:b5,dpup:b13,dpleft:b11,dpdown:b14,dpright:b12,leftx:a0,lefty:a1,rightx:a3,righty:a4,lefttrigger:a2,righttrigger:a5,",
	"030000008916000001fd000024010000,Razer Onza Classic Edition,x:b2,a:b0,b:b1,y:b3,back:b6,guide:b8,start:b7,dpleft:b11,dpdown:b14,dpright:b12,dpup:b13,leftshoulder:b4,lefttrigger:a2,rightshoulder:b5,righttrigger:a5,leftstick:b9,rightstick:b10,leftx:a0,lefty:a1,rightx:a3,righty:a4,",
	"030000008f0e00000300000010010000,GreenAsia Inc. USB Joystick,x:b3,a:b2,b:b1,y:b0,back:b8,start:b9,dpleft:h0.8,dpdown:h0.0,dpdown:h0.4,dpright:h0.0,dpright:h0.2,dpup:h0.0,dpup:h0.1,leftshoulder:h0.0,leftshoulder:b6,lefttrigger:b4,rightshoulder:b7,righttrigger:b5,leftstick:b10,rightstick:b11,leftx:a0,lefty:a1,rightx:a3,righty:a2,",
	"030000008f0e00001200000010010000,GreenAsia Inc. USB Joystick,x:b2,a:b0,b:b1,y:b3,back:b8,start:b9,dpleft:h0.8,dpdown:h0.4,dpright:h0.2,dpup:h0.1,leftshoulder:b4,lefttrigger:b5,rightshoulder:b6,righttrigger:b7,leftstick:b10,rightstick:b11,leftx:a0,lefty:a1,rightx:a3,righty:a2,",
	"03000000a30600000901000000010000,Saitek P880,a:b2,b:b3,y:b1,x:b0,leftstick:b8,rightstick:b9,leftshoulder:b4,rightshoulder:b5,dpup:h0.1,dpleft:h0.8,dpdown:h0.4,dpright:h0.2,leftx:a0,lefty:a1,rightx:a3,righty:a2,lefttrigger:b6,righttrigger:b7,",
	"03000000a30600000c04000011010000,Saitek P2900 Wireless Pad,a:b1,b:b2,y:b3,x:b0,start:b12,guide:b9,back:b8,leftstick:b10,rightstick:b11,leftshoulder:b6,rightshoulder:b7,dpup:h0.1,dpleft:h0.8,dpdown:h0.4,dpright:h0.2,leftx:a0,lefty:a1,rightx:a3,righty:a2,lefttrigger:b4,righttrigger:b5,",
	"03000000a306000018f5000010010000,Saitek PLC Saitek P3200 Rumble Pad,x:b0,a:b1,b:b2,y:b3,back:b8,start:b9,dpleft:h0.8,dpdown:h0.0,dpdown:h0.4,dpright:h0.0,dpright:h0.2,dpup:h0.0,dpup:h0.1,leftshoulder:h0.0,leftshoulder:b4,lefttrigger:a2,rightshoulder:b6,rightshoulder:b5,righttrigger:b7,leftstick:b10,rightstick:b11,leftx:a0,lefty:a1,rightx:a3,righty:a4,",
	"03000000a306000023f6000011010000,Saitek Cyborg V.1 Game Pad,a:b1,b:b2,y:b3,x:b0,start:b9,guide:b12,back:b8,leftstick:b10,rightstick:b11,leftshoulder:b4,rightshoulder:b5,dpup:h0.1,dpleft:h0.8,dpdown:h0.4,dpright:h0.2,leftx:a0,lefty:a1,rightx:a2,righty:a4,lefttrigger:b6,righttrigger:b7,",
	"03000000ad1b000001f5000033050000,Hori Pad EX Turbo 2,a:b0,b:b1,y:b3,x:b2,start:b7,guide:b8,back:b6,leftstick:b9,rightstick:b10,leftshoulder:b4,rightshoulder:b5,dpup:h0.1,dpleft:h0.8,dpdown:h0.4,dpright:h0.2,leftx:a0,lefty:a1,rightx:a3,righty:a4,lefttrigger:a2,righttrigger:a5,",
	"03000000ad1b000016f0000090040000,Mad Catz Xbox 360 Controller,a:b0,b:b1,y:b3,x:b2,start:b7,guide:b8,back:b6,leftstick:b9,rightstick:b10,leftshoulder:b4,rightshoulder:b5,dpup:h0.1,dpleft:h0.8,dpdown:h0.4,dpright:h0.2,leftx:a0,lefty:a1,rightx:a3,righty:a4,lefttrigger:a2,righttrigger:a5,",
	"03000000ad1b00002ef0000090040000,Mad Catz Fightpad SFxT,a:b0,b:b1,y:b3,x:b2,start:b7,guide:b8,back:b6,leftshoulder:b4,rightshoulder:b5,dpup:h0.1,dpleft:h0.8,dpdown:h0.4,dpright:h0.2,lefttrigger:a2,righttrigger:a5,",
	"03000000ba2200002010000001010000,Jess Technology USB Game Controller,a:b2,b:b1,back:b8,dpdown:h0.4,dpleft:h0.8,dpright:h0.2,dpup:h0.1,guide:,leftshoulder:b4,lefttrigger:b6,leftx:a0,lefty:a1,rightshoulder:b5,righttrigger:b7,rightx:a3,righty:a2,start:b9,x:b3,y:b0,",
	"03000000bd12000015d0000010010000,Tomee SNES USB Controller,x:b3,a:b2,b:b1,y:b0,back:b8,start:b9,leftshoulder:b4,rightshoulder:b5,leftx:a0,lefty:a1,",
	"03000000c9110000f055000011010000,HJC Game GAMEPAD,platform:Linux,x:b2,a:b0,b:b1,y:b3,back:b4,back:b8,start:b9,dpleft:h0.8,dpdown:h0.0,dpdown:h0.4,dpright:h0.0,dpright:h0.2,dpup:h0.0,dpup:h0.1,leftshoulder:h0.0,leftshoulder:b4,lefttrigger:b6,rightshoulder:b5,righttrigger:b7,leftstick:b10,rightstick:b11,leftx:a0,lefty:a1,rightx:a2,righty:a3,",
	"03000000d814000007cd000011010000,Toodles 2008 Chimp PC/PS3,a:b0,b:b1,y:b2,x:b3,start:b9,back:b8,leftshoulder:b4,rightshoulder:b5,leftx:a0,lefty:a1,lefttrigger:b6,righttrigger:b7,",
	"03000000d81400000862000011010000,HitBox (PS3/PC) Analog Mode,a:b1,b:b2,y:b3,x:b0,start:b12,guide:b9,back:b8,leftshoulder:b4,rightshoulder:b5,lefttrigger:b6,righttrigger:b7,leftx:a0,lefty:a1,",
	"03000000de280000ff11000001000000,Valve Streaming Gamepad,a:b0,b:b1,back:b6,dpdown:h0.4,dpleft:h0.8,dpright:h0.2,dpup:h0.1,guide:b8,leftshoulder:b4,leftstick:b9,lefttrigger:a2,leftx:a0,lefty:a1,rightshoulder:b5,rightstick:b10,righttrigger:a5,rightx:a3,righty:a4,start:b7,x:b2,y:b3,",
	"03000000f0250000c183000010010000,Goodbetterbest Ltd USB Controller,x:b0,a:b1,b:b2,y:b3,back:b8,guide:b12,start:b9,dpleft:h0.8,dpdown:h0.0,dpdown:h0.4,dpright:h0.0,dpright:h0.2,dpup:h0.0,dpup:h0.1,leftshoulder:h0.0,leftshoulder:b4,lefttrigger:b6,rightshoulder:b5,righttrigger:b7,leftstick:b10,rightstick:b11,leftx:a0,lefty:a1,rightx:a2,righty:a3,",
	"03000000fd0500000030000000010000,InterAct GoPad I-73000 (Fighting Game Layout),a:b3,b:b4,y:b1,x:b0,start:b7,back:b6,leftx:a0,lefty:a1,rightshoulder:b2,righttrigger:b5,",
	"03000000fd0500002a26000000010000,3dfx InterAct HammerHead FX,leftx:a0,lefty:a1,dpdown:h0.4,rightstick:b5,rightshoulder:b7,rightx:a2,start:b11,righty:a3,dpleft:h0.8,lefttrigger:b8,x:b0,dpup:h0.1,back:b10,leftstick:b2,leftshoulder:b6,y:b1,a:b3,dpright:h0.2,righttrigger:b9,b:b4,",
	"03000000ff1100003133000010010000,PC Game Controller,a:b2,b:b1,y:b0,x:b3,start:b9,back:b8,leftstick:b10,rightstick:b11,leftshoulder:b4,rightshoulder:b5,dpup:h0.1,dpleft:h0.8,dpdown:h0.4,dpright:h0.2,leftx:a0,lefty:a1,rightx:a2,righty:a3,lefttrigger:b6,righttrigger:b7,",
	"05000000010000000100000003000000,Nintendo Wiimote,a:b0,b:b1,y:b3,x:b2,start:b9,guide:b10,back:b8,leftstick:b11,rightstick:b12,leftshoulder:b4,rightshoulder:b5,dpup:h0.1,dpleft:h0.8,dpdown:h0.4,dpright:h0.2,leftx:a0,lefty:a1,rightx:a2,righty:a3,lefttrigger:b6,righttrigger:b7,",
	"05000000102800000900000000010000,8Bitdo SFC30 GamePad,x:b4,a:b1,b:b0,y:b3,back:b10,start:b11,leftshoulder:b6,rightshoulder:b7,leftx:a0,lefty:a1,",
	"05000000362800000100000002010000,OUYA Game Controller,a:b0,b:b3,dpdown:b9,dpleft:b10,dpright:b11,dpup:b8,guide:b14,leftshoulder:b4,leftstick:b6,lefttrigger:a2,leftx:a0,lefty:a1,rightshoulder:b5,rightstick:b7,righttrigger:a5,rightx:a3,righty:a4,x:b1,y:b2,",
	"05000000362800000100000003010000,OUYA Game Controller,a:b0,b:b3,dpdown:b9,dpleft:b10,dpright:b11,dpup:b8,guide:b14,leftshoulder:b4,leftstick:b6,lefttrigger:a2,leftx:a0,lefty:a1,rightshoulder:b5,rightstick:b7,righttrigger:a5,rightx:a3,righty:a4,x:b1,y:b2,",
	"05000000362800000100000004010000,OUYA Game Controller,leftx:a0,lefty:a1,dpdown:b9,rightstick:b7,rightshoulder:b5,rightx:a3,start:b16,righty:a4,dpleft:b10,lefttrigger:b12,x:b1,dpup:b8,back:b14,leftstick:b6,leftshoulder:b4,y:b2,a:b0,dpright:b11,righttrigger:b13,b:b3,",
	"05000000380700006652000025010000,Mad Catz C.T.R.L.R ,x:b0,a:b1,b:b2,y:b3,back:b8,guide:b12,start:b9,dpleft:h0.8,dpdown:h0.4,dpright:h0.2,dpup:h0.1,leftshoulder:b4,lefttrigger:b6,rightshoulder:b5,righttrigger:b7,leftstick:b10,rightstick:b11,leftx:a0,lefty:a1,rightx:a2,righty:a3,",
	"0500000047532047616d657061640000,GameStop Gamepad,a:b0,b:b1,back:b8,dpdown:h0.4,dpleft:h0.8,dpright:h0.2,dpup:h0.1,guide:,leftshoulder:b4,leftstick:b10,lefttrigger:b6,leftx:a0,lefty:a1,rightshoulder:b5,rightstick:b11,righttrigger:b7,rightx:a2,righty:a3,start:b9,x:b2,y:b3,",
	"050000004c0500006802000000010000,PS3 Controller (Bluetooth),a:b14,b:b13,y:b12,x:b15,start:b3,guide:b16,back:b0,leftstick:b1,rightstick:b2,leftshoulder:b10,rightshoulder:b11,dpup:b4,dpleft:b7,dpdown:b6,dpright:b5,leftx:a0,lefty:a1,rightx:a2,righty:a3,lefttrigger:b8,righttrigger:b9,",
	"050000004c050000c405000000010000,PS4 Controller (Bluetooth),a:b1,b:b2,back:b8,dpdown:h0.4,dpleft:h0.8,dpright:h0.2,dpup:h0.1,guide:b12,leftshoulder:b4,leftstick:b10,lefttrigger:a3,leftx:a0,lefty:a1,rightshoulder:b5,rightstick:b11,righttrigger:a4,rightx:a2,righty:a5,start:b9,x:b0,y:b3,",
	"050000004c050000cc09000000010000,Sony DualShock 4 V2 BT,a:b1,b:b2,y:b3,x:b0,start:b9,guide:b12,back:b13,leftstick:b10,rightstick:b11,leftshoulder:b4,rightshoulder:b5,dpup:h0.1,dpleft:h0.8,dpdown:h0.4,dpright:h0.2,leftx:a0,lefty:a1,rightx:a2,righty:a5,lefttrigger:a3,righttrigger:a4,",
	"050000007e0500003003000001000000,Nintendo Wii U Pro Controller,a:b0,b:b1,x:b3,y:b2,back:b8,start:b9,guide:b10,leftshoulder:b4,rightshoulder:b5,leftstick:b11,rightstick:b12,leftx:a0,lefty:a1,rightx:a2,righty:a3,lefttrigger:b6,righttrigger:b7,dpup:b13,dpleft:b15,dpdown:b14,dpright:b16,",
	"05000000a00500003232000001000000,8Bitdo Zero GamePad,a:b0,b:b1,x:b3,y:b4,back:b10,start:b11,leftshoulder:b6,rightshoulder:b7,leftx:a0,lefty:a1,",
	"05000000ac0500003232000001000000,VR-BOX,a:b0,b:b1,x:b2,y:b3,start:b9,back:b8,leftstick:b10,rightstick:b11,leftshoulder:b6,rightshoulder:b7,dpup:h0.1,dpleft:h0.8,dpdown:h0.4,dpright:h0.2,leftx:a0,lefty:a1,rightx:a3,righty:a2,lefttrigger:b4,righttrigger:b5,",
	"05000000d6200000ad0d000001000000,Moga Pro,a:b0,b:b1,y:b3,x:b2,start:b6,leftstick:b7,rightstick:b8,leftshoulder:b4,rightshoulder:b5,dpup:h0.1,dpleft:h0.8,dpdown:h0.4,dpright:h0.2,leftx:a0,lefty:a1,rightx:a2,righty:a3,lefttrigger:a5,righttrigger:a4,",
	"060000004c0500006802000000010000,PS3 Controller (Bluetooth),a:b14,b:b13,y:b12,x:b15,start:b3,guide:b16,back:b0,leftstick:b1,rightstick:b2,leftshoulder:b10,rightshoulder:b11,dpup:b4,dpleft:b7,dpdown:b6,dpright:b5,leftx:a0,lefty:a1,rightx:a2,righty:a3,lefttrigger:b8,righttrigger:b9,",
#endif

#if defined(__ANDROID__)
	"Default Android Gamepad,Default Controller,leftx:a0,lefty:a1,dpdown:h0.4,rightstick:b8,rightshoulder:b10,rightx:a2,start:b6,righty:a3,dpleft:h0.8,lefttrigger:a4,x:b2,dpup:h0.1,back:b4,leftstick:b7,leftshoulder:b9,y:b3,a:b0,dpright:h0.2,righttrigger:a5,b:b1,",
	"47656e6572696320582d426f78207061,Logitech F-310,leftx:a0,lefty:a1,dpdown:h0.4,rightstick:b8,rightshoulder:b10,rightx:a2,start:b6,righty:a3,dpleft:h0.8,lefttrigger:a5,x:b2,dpup:h0.1,leftstick:b7,leftshoulder:b9,y:b3,a:b0,dpright:h0.2,righttrigger:a4,b:b1,",
	"484f524920434f2e2c4c544420205041,Hori Gem Pad 3,leftx:a0,lefty:a1,dpdown:h0.4,rightstick:b6,rightshoulder:b18,rightx:a2,start:b16,righty:a3,dpleft:h0.8,lefttrigger:b9,x:b0,dpup:h0.1,back:b15,leftstick:b4,leftshoulder:b3,y:b2,a:b1,dpright:h0.2,righttrigger:b10,b:b17,",
	"4d6963726f736f667420582d426f7820,Microsoft X-Box 360 pad,leftx:a0,lefty:a1,dpdown:h0.4,rightstick:b8,rightshoulder:b10,rightx:a2,start:b6,righty:a3,dpleft:h0.8,lefttrigger:a4,x:b2,dpup:h0.1,leftstick:b7,leftshoulder:b9,y:b3,a:b0,dpright:h0.2,righttrigger:a5,b:b1,",
	"4e564944494120436f72706f72617469,NVIDIA Controller,a:b0,b:b1,dpdown:h0.4,dpleft:h0.8,dpright:h0.2,dpup:h0.1,leftshoulder:b9,leftstick:b7,lefttrigger:a4,leftx:a0,lefty:a1,rightshoulder:b10,rightstick:b8,righttrigger:a5,rightx:a2,righty:a3,start:b6,x:b2,y:b3,",
	"532e542e442e20496e74657261637420,3dfx InterAct HammerHead FX,leftx:a0,lefty:a1,dpdown:h0.4,rightstick:b25,rightshoulder:b27,rightx:a2,start:b31,righty:a3,dpleft:h0.8,lefttrigger:b28,x:b20,dpup:h0.1,back:b30,leftstick:b22,leftshoulder:b26,y:b21,a:b23,dpright:h0.2,righttrigger:b29,b:b24,",
	"506572666f726d616e63652044657369,PDP Rock Candy Wireless Controller for PS3,leftx:a0,lefty:a1,dpdown:h0.4,rightstick:b6,rightshoulder:b18,rightx:a2,start:b16,righty:a3,dpleft:h0.8,lefttrigger:b9,x:b0,dpup:h0.1,back:h0.2,leftstick:b4,leftshoulder:b3,y:b2,a:b1,dpright:h0.2,righttrigger:b10,b:b17,",
	"4f5559412047616d6520436f6e74726f,OUYA Game Controller,leftx:a1,lefty:a3,dpdown:b12,rightstick:b8,rightshoulder:b10,rightx:a6,start:b-86,righty:a7,dpleft:b13,lefttrigger:b15,x:b2,dpup:b11,leftstick:b7,leftshoulder:b9,y:b3,a:b0,dpright:b14,righttrigger:b16,b:b1,",
#endif

#ifdef JAVASCRIPT_ENABLED
	"Default HTML5 Gamepad, Default Mapping,leftx:a0,lefty:a1,dpdown:b13,rightstick:b11,rightshoulder:b5,rightx:a2,start:b9,righty:a3,dpleft:b14,lefttrigger:a6,x:b2,dpup:b12,back:b8,leftstick:b10,leftshoulder:b4,y:b3,a:b0,dpright:b15,righttrigger:a7,b:b1,",
	"303435652d303238652d4d6963726f73,Wired X360 Controller,leftx:a0,lefty:a1,dpdown:a7,rightstick:b10,rightshoulder:b5,rightx:a3,start:b7,righty:a4,dpleft:a6,lefttrigger:a2,x:b2,dpup:a7,back:b6,leftstick:b9,leftshoulder:b4,y:b3,a:b0,dpright:a6,righttrigger:a5,b:b1,",
	"303435652d303731392d58626f782033,Wireless X360 Controller,leftx:a0,lefty:a1,dpdown:b14,rightstick:b10,rightshoulder:b5,rightx:a3,start:b7,righty:a4,dpleft:b11,lefttrigger:a2,x:b2,dpup:b13,back:b6,leftstick:b9,leftshoulder:b4,y:b3,a:b0,dpright:b12,righttrigger:a5,b:b1,",
	"303534632d303236382d536f6e792050,PS3 Controller USB/Linux,leftx:a0,lefty:a1,dpdown:b6,rightstick:b2,rightshoulder:b11,rightx:a2,start:b3,righty:a3,dpleft:b7,lefttrigger:b8,x:b15,dpup:b4,back:b0,leftstick:b1,leftshoulder:b10,y:b12,a:b14,dpright:b5,righttrigger:b9,b:b13,",
	"303534632d303563342d536f6e792043,PS4 Controller USB/Linux,leftx:a0,lefty:a1,dpdown:a7,rightstick:b11,rightshoulder:b5,rightx:a2,start:b9,righty:a5,dpleft:a6,lefttrigger:a3,x:b0,dpup:a7,back:b8,leftstick:b10,leftshoulder:b4,y:b3,a:b1,dpright:a6,righttrigger:a4,b:b2,",
	"303534632d303563342d576972656c65,PS4 Controller USB/Win,leftx:a0,lefty:a1,dpdown:b15,rightstick:b11,rightshoulder:b5,rightx:a2,start:b9,righty:a5,lefttrigger:a3,x:b0,dpup:b14,dpleft:b16,dpright:b17,back:b8,leftstick:b10,leftshoulder:b4,y:b3,a:b1,righttrigger:b7,b:b2,",
	"c2a94d6963726f736f66742058626f78,Wireless X360 Controller,leftx:a0,lefty:a1,dpdown:b14,rightstick:b10,rightshoulder:b5,rightx:a3,start:b7,righty:a4,dpleft:b11,lefttrigger:a2,x:b2,dpup:b13,back:b6,leftstick:b9,leftshoulder:b4,y:b3,a:b0,dpright:b12,righttrigger:a5,b:b1,",
#endif

#ifdef UWP_ENABLED
	"__UWP_GAMEPAD__,Xbox Controller,a:b2,b:b3,x:b4,y:b5,start:b0,back:b1,leftstick:b12,rightstick:b13,leftshoulder:b10,rightshoulder:b11,dpup:b6,dpdown:b7,dpleft:b8,dpright:b9,leftx:a0,lefty:a1,rightx:a2,righty:a3,lefttrigger:a4,righttrigger:a5,",
#endif
	NULL
};

InputDefault::InputDefault() {

	mouse_button_mask = 0;
	emulate_touch = false;
	main_loop = NULL;

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

	String env_mapping = OS::get_singleton()->get_environment("SDL_GAMECONTROLLERCONFIG");
	if (env_mapping != "") {

		Vector<String> entries = env_mapping.split("\n");
		for (int i = 0; i < entries.size(); i++) {
			if (entries[i] == "")
				continue;
			parse_mapping(entries[i]);
		};
	};

	int i = 0;
	while (s_ControllerMappings[i]) {

		parse_mapping(s_ControllerMappings[i++]);
	};
}

void InputDefault::joy_button(int p_device, int p_button, bool p_pressed) {

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
	};

	Map<int, JoyEvent>::Element *el = map_db[joy.mapping].buttons.find(p_button);
	if (!el) {
		//don't process un-mapped events for now, it could mess things up badly for devices with additional buttons/axis
		//return _button_event(p_last_id, p_device, p_button, p_pressed);
		return;
	};

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
	};

	if (map.type == TYPE_AXIS) {
		_axis_event(p_device, map.index, p_pressed ? 1.0 : 0.0);
	};

	return; // no event?
};

void InputDefault::joy_axis(int p_device, int p_axis, const JoyAxis &p_value) {

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

	Map<int, JoyEvent>::Element *el = map_db[joy.mapping].axis.find(p_axis);
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
		bool pressed = p_value.value > deadzone ? true : false;
		if (pressed == joy_buttons_pressed.has(_combine_device(map.index, p_device))) {
			// button already pressed or released, this is an axis bounce value
			return;
		};
		_button_event(p_device, map.index, pressed);
		return;
	};

	if (map.type == TYPE_AXIS) {

		_axis_event(p_device, map.index, val);
		return;
	};
	//printf("invalid mapping\n");
	return;
};

void InputDefault::joy_hat(int p_device, int p_val) {

	_THREAD_SAFE_METHOD_;
	const Joypad &joy = joy_names[p_device];

	JoyEvent *map;

	if (joy.mapping == -1) {
		map = hat_map_default;
	} else {
		map = map_db[joy.mapping].hat;
	};

	int cur_val = joy_names[p_device].hat_current;

	if ((p_val & HAT_MASK_UP) != (cur_val & HAT_MASK_UP)) {
		_button_event(p_device, map[HAT_UP].index, p_val & HAT_MASK_UP);
	};

	if ((p_val & HAT_MASK_RIGHT) != (cur_val & HAT_MASK_RIGHT)) {
		_button_event(p_device, map[HAT_RIGHT].index, p_val & HAT_MASK_RIGHT);
	};
	if ((p_val & HAT_MASK_DOWN) != (cur_val & HAT_MASK_DOWN)) {
		_button_event(p_device, map[HAT_DOWN].index, p_val & HAT_MASK_DOWN);
	};
	if ((p_val & HAT_MASK_LEFT) != (cur_val & HAT_MASK_LEFT)) {
		_button_event(p_device, map[HAT_LEFT].index, p_val & HAT_MASK_LEFT);
	};

	joy_names[p_device].hat_current = p_val;
};

void InputDefault::_button_event(int p_device, int p_index, bool p_pressed) {

	Ref<InputEventJoypadButton> ievent;
	ievent.instance();
	ievent->set_device(p_device);
	ievent->set_button_index(p_index);
	ievent->set_pressed(p_pressed);

	parse_input_event(ievent);
};

void InputDefault::_axis_event(int p_device, int p_axis, float p_value) {

	Ref<InputEventJoypadMotion> ievent;
	ievent.instance();
	ievent->set_device(p_device);
	ievent->set_axis(p_axis);
	ievent->set_axis_value(p_value);

	parse_input_event(ievent);
};

InputDefault::JoyEvent InputDefault::_find_to_event(String p_to) {

	// string names of the SDL buttons in the same order as input_event.h godot buttons
	static const char *buttons[] = { "a", "b", "x", "y", "leftshoulder", "rightshoulder", "lefttrigger", "righttrigger", "leftstick", "rightstick", "back", "start", "dpup", "dpdown", "dpleft", "dpright", "guide", NULL };

	static const char *axis[] = { "leftx", "lefty", "rightx", "righty", NULL };

	JoyEvent ret;
	ret.type = -1;

	int i = 0;
	while (buttons[i]) {

		if (p_to == buttons[i]) {
			//printf("mapping button %s\n", buttons[i]);
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

void InputDefault::parse_mapping(String p_mapping) {

	_THREAD_SAFE_METHOD_;
	JoyDeviceMapping mapping;
	for (int i = 0; i < HAT_MAX; ++i)
		mapping.hat[i].index = 1024 + i;

	Vector<String> entry = p_mapping.split(",");
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

void InputDefault::add_joy_mapping(String p_mapping, bool p_update_existing) {
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

void InputDefault::remove_joy_mapping(String p_guid) {
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

void InputDefault::set_fallback_mapping(String p_guid) {

	for (int i = 0; i < map_db.size(); i++) {
		if (map_db[i].uid == p_guid) {
			fallback_mapping = i;
			return;
		}
	}
}

//Defaults to simple implementation for platforms with a fixed gamepad layout, like consoles.
bool InputDefault::is_joy_known(int p_device) {

	return OS::get_singleton()->is_joy_known(p_device);
}

String InputDefault::get_joy_guid(int p_device) const {
	return OS::get_singleton()->get_joy_guid(p_device);
}

//platforms that use the remapping system can override and call to these ones
bool InputDefault::is_joy_mapped(int p_device) {
	int mapping = joy_names[p_device].mapping;
	return mapping != -1 ? (mapping != fallback_mapping) : false;
}

String InputDefault::get_joy_guid_remapped(int p_device) const {
	ERR_FAIL_COND_V(!joy_names.has(p_device), "");
	return joy_names[p_device].uid;
}

Array InputDefault::get_connected_joypads() {
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

static const char *_buttons[] = {
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

static const char *_axes[] = {
	"Left Stick X",
	"Left Stick Y",
	"Right Stick X",
	"Right Stick Y",
	"",
	"",
	"L2",
	"R2"
};

String InputDefault::get_joy_button_string(int p_button) {
	ERR_FAIL_INDEX_V(p_button, JOY_BUTTON_MAX, "");
	return _buttons[p_button];
}

int InputDefault::get_joy_button_index_from_string(String p_button) {
	for (int i = 0; i < JOY_BUTTON_MAX; i++) {
		if (p_button == _buttons[i]) {
			return i;
		}
	}
	ERR_FAIL_V(-1);
}

int InputDefault::get_unused_joy_id() {
	for (int i = 0; i < JOYPADS_MAX; i++) {
		if (!joy_names.has(i) || !joy_names[i].connected) {
			return i;
		}
	}
	return -1;
}

String InputDefault::get_joy_axis_string(int p_axis) {
	ERR_FAIL_INDEX_V(p_axis, JOY_AXIS_MAX, "");
	return _axes[p_axis];
}

int InputDefault::get_joy_axis_index_from_string(String p_axis) {
	for (int i = 0; i < JOY_AXIS_MAX; i++) {
		if (p_axis == _axes[i]) {
			return i;
		}
	}
	ERR_FAIL_V(-1);
}
