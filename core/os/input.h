/*************************************************************************/
/*  input.h                                                              */
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
#ifndef INPUT_H
#define INPUT_H

#include "object.h"
#include "os/main_loop.h"
#include "os/thread_safe.h"

class Input : public Object {

	GDCLASS(Input, Object);

	static Input *singleton;

protected:
	static void _bind_methods();

public:
	enum MouseMode {
		MOUSE_MODE_VISIBLE,
		MOUSE_MODE_HIDDEN,
		MOUSE_MODE_CAPTURED,
		MOUSE_MODE_CONFINED
	};

	void set_mouse_mode(MouseMode p_mode);
	MouseMode get_mouse_mode() const;

	static Input *get_singleton();

	virtual bool is_key_pressed(int p_scancode) const = 0;
	virtual bool is_mouse_button_pressed(int p_button) const = 0;
	virtual bool is_joy_button_pressed(int p_device, int p_button) const = 0;
	virtual bool is_action_pressed(const StringName &p_action) const = 0;
	virtual bool is_action_just_pressed(const StringName &p_action) const = 0;
	virtual bool is_action_just_released(const StringName &p_action) const = 0;

	virtual float get_joy_axis(int p_device, int p_axis) const = 0;
	virtual String get_joy_name(int p_idx) = 0;
	virtual Array get_connected_joypads() = 0;
	virtual void joy_connection_changed(int p_idx, bool p_connected, String p_name, String p_guid) = 0;
	virtual void add_joy_mapping(String p_mapping, bool p_update_existing = false) = 0;
	virtual void remove_joy_mapping(String p_guid) = 0;
	virtual bool is_joy_known(int p_device) = 0;
	virtual String get_joy_guid(int p_device) const = 0;
	virtual Vector2 get_joy_vibration_strength(int p_device) = 0;
	virtual float get_joy_vibration_duration(int p_device) = 0;
	virtual uint64_t get_joy_vibration_timestamp(int p_device) = 0;
	virtual void start_joy_vibration(int p_device, float p_weak_magnitude, float p_strong_magnitude, float p_duration = 0) = 0;
	virtual void stop_joy_vibration(int p_device) = 0;

	virtual Point2 get_mouse_pos() const = 0;
	virtual Point2 get_last_mouse_speed() const = 0;
	virtual int get_mouse_button_mask() const = 0;

	virtual void warp_mouse_pos(const Vector2 &p_to) = 0;
	virtual Point2i warp_mouse_motion(const InputEventMouseMotion &p_motion, const Rect2 &p_rect) = 0;

	virtual Vector3 get_gravity() const = 0;
	virtual Vector3 get_accelerometer() const = 0;
	virtual Vector3 get_magnetometer() const = 0;
	virtual Vector3 get_gyroscope() const = 0;

	virtual void action_press(const StringName &p_action) = 0;
	virtual void action_release(const StringName &p_action) = 0;

	void get_argument_options(const StringName &p_function, int p_idx, List<String> *r_options) const;

	virtual bool is_emulating_touchscreen() const = 0;

	virtual void set_custom_mouse_cursor(const RES &p_cursor, const Vector2 &p_hotspot = Vector2()) = 0;
	virtual void set_mouse_in_window(bool p_in_window) = 0;

	virtual String get_joy_button_string(int p_button) = 0;
	virtual String get_joy_axis_string(int p_axis) = 0;
	virtual int get_joy_button_index_from_string(String p_button) = 0;
	virtual int get_joy_axis_index_from_string(String p_axis) = 0;

	virtual void parse_input_event(const InputEvent &p_event) = 0;

	Input();
};

VARIANT_ENUM_CAST(Input::MouseMode);

#endif // INPUT_H
