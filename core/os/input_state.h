/*************************************************************************/
/*  input_state.h                                                        */
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

#ifndef INPUT_STATE_H
#define INPUT_STATE_H

#include "core/resource.h"

class InputState : public Resource {
	GDCLASS(InputState, Resource);

	friend class InputDefault;

	uint64_t physics_frame;
	uint64_t idle_frame;

	Set<int> keys_pressed;
	Set<int> joy_buttons_pressed;
	Map<int, float> joy_axis;
	int mouse_button_mask;

	struct VibrationInfo {
		float weak_magnitude;
		float strong_magnitude;
		float duration; // Duration in seconds
		uint64_t timestamp;
	};
	Map<int, VibrationInfo> joy_vibration;

	Vector3 gravity;
	Vector3 accelerometer;
	Vector3 magnetometer;
	Vector3 gyroscope;
	Vector2 mouse_pos;

	struct Action {
		bool pressed;
		bool exact;
		float strength;
		uint64_t physics_frame;
		uint64_t idle_frame;

		Action() :
				pressed(false),
				exact(false),
				strength(0.0f),
				physics_frame(0),
				idle_frame(0) {}
	};
	Map<StringName, Action> action_state;

	void _set_data(const Dictionary &p_data);
	Dictionary _get_data() const;

protected:
	static void _bind_methods();

public:
	bool is_key_pressed(int p_scancode) const;
	bool is_mouse_button_pressed(int p_button) const;
	bool is_joy_button_pressed(int p_device, int p_button) const;
	bool is_action_pressed(const StringName &p_action) const;
	bool is_action_just_pressed(const StringName &p_action) const;
	bool is_action_just_released(const StringName &p_action) const;
	float get_action_strength(const StringName &p_action) const;

	float get_joy_axis(int p_device, int p_axis) const;
	Vector2 get_joy_vibration_strength(int p_device);
	float get_joy_vibration_duration(int p_device);
	uint64_t get_joy_vibration_timestamp(int p_device);

	void start_joy_vibration(int p_device, float p_weak_magnitude, float p_strong_magnitude, float p_duration = 0);
	void stop_joy_vibration(int p_device);

	Vector3 get_gravity() const;
	Vector3 get_accelerometer() const;
	Vector3 get_magnetometer() const;
	Vector3 get_gyroscope() const;

	Point2 get_mouse_position() const;
	// Point2 get_last_mouse_speed() const;
	int get_mouse_button_mask() const;

	void set_gravity(const Vector3 &p_gravity);
	void set_accelerometer(const Vector3 &p_accel);
	void set_magnetometer(const Vector3 &p_magnetometer);
	void set_gyroscope(const Vector3 &p_gyroscope);
	void set_joy_axis(int p_device, int p_axis, float p_value);

	void set_mouse_position(const Point2 &p_posf);

	void action_press(const StringName &p_action, float p_strength = 1.f);
	void action_release(const StringName &p_action);

	void release_pressed_events();

	void feed(const Ref<InputState> &p_state);
	void clear();

	uint64_t get_physics_frame() const;
	uint64_t get_idle_frame() const;

	InputState();
};

#endif // INPUT_STATE_H
