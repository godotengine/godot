/*************************************************************************/
/*  input.h                                                              */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                 */
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
#include "os/thread_safe.h"
#include "os/main_loop.h"

class Input : public Object {

	OBJ_TYPE( Input, Object );

	static Input *singleton;

protected:

	static void _bind_methods();
public:

	enum MouseMode {
		MOUSE_MODE_VISIBLE,
		MOUSE_MODE_HIDDEN,
		MOUSE_MODE_CAPTURED
	};

	void set_mouse_mode(MouseMode p_mode);
	MouseMode get_mouse_mode() const;

	static Input *get_singleton();

	virtual bool is_key_pressed(int p_scancode)=0;	
	virtual bool is_mouse_button_pressed(int p_button)=0;
	virtual bool is_joy_button_pressed(int p_device, int p_button)=0;
	virtual bool is_action_pressed(const StringName& p_action)=0;

	virtual float get_joy_axis(int p_device,int p_axis)=0;
	virtual String get_joy_name(int p_idx)=0;
	virtual void joy_connection_changed(int p_idx, bool p_connected, String p_name, String p_guid = "")=0;


	virtual Point2 get_mouse_pos() const=0;
	virtual Point2 get_mouse_speed() const=0;
	virtual int get_mouse_button_mask() const=0;

	virtual void warp_mouse_pos(const Vector2& p_to)=0;

	virtual Vector3 get_accelerometer()=0;

	virtual void action_press(const StringName& p_action)=0;
	virtual void action_release(const StringName& p_action)=0;

	void get_argument_options(const StringName& p_function,int p_idx,List<String>*r_options) const;


	Input();
};

VARIANT_ENUM_CAST(Input::MouseMode);

class InputDefault : public Input {

	OBJ_TYPE( InputDefault, Input );

protected:
	_THREAD_SAFE_CLASS_

	int mouse_button_mask;
	Set<int> keys_pressed;
	Set<int> joy_buttons_pressed;
	Map<int,float> joy_axis;
	Map<StringName,int> custom_action_press;
	Vector3 accelerometer;
	Vector2 mouse_pos;
	MainLoop *main_loop;

	struct SpeedTrack {

		uint64_t last_tick;
		Vector2 speed;
		Vector2 accum;
		float accum_t;
		float min_ref_frame;
		float max_ref_frame;

		void update(const Vector2& p_delta_p);
		void reset();
		SpeedTrack();
	};

	SpeedTrack mouse_speed_track;

	struct Joystick {
		StringName name;
		StringName uid;
		int mapping;
		int hat_current;
	};

	Map<int, Joystick> joy_names;

public:

	virtual bool is_key_pressed(int p_scancode);
	virtual bool is_mouse_button_pressed(int p_button);
	virtual bool is_joy_button_pressed(int p_device, int p_button);
	virtual bool is_action_pressed(const StringName& p_action);

	virtual float get_joy_axis(int p_device,int p_axis);

	String get_joy_name(int p_idx);
	void joy_connection_changed(int p_idx, bool p_connected, String p_name, String p_guid="");

	virtual Vector3 get_accelerometer();

	virtual Point2 get_mouse_pos() const;
	virtual Point2 get_mouse_speed() const;
	virtual int get_mouse_button_mask() const;

	virtual void warp_mouse_pos(const Vector2& p_to);

	void parse_input_event(const InputEvent& p_event);
	void set_accelerometer(const Vector3& p_accel);
	void set_joy_axis(int p_device,int p_axis,float p_value);

	void set_main_loop(MainLoop *main_loop);
	void set_mouse_pos(const Point2& p_posf);

	void action_press(const StringName& p_action);
	void action_release(const StringName& p_action);

	void iteration(float p_step);

	InputDefault();

};

class InputPC : public InputDefault {

	OBJ_TYPE( InputPC, InputDefault );

public:
	enum HatMask {
		HAT_MASK_CENTER = 0,
		HAT_MASK_UP = 1,
		HAT_MASK_RIGHT = 2,
		HAT_MASK_DOWN = 4,
		HAT_MASK_LEFT = 8,
	};

	enum HatDir {
		HAT_UP,
		HAT_RIGHT,
		HAT_DOWN,
		HAT_LEFT,
		HAT_MAX,
	};

private:

	enum JoyType {
		TYPE_BUTTON,
		TYPE_AXIS,
		TYPE_HAT,
		TYPE_MAX,
	};

	struct JoyEvent {
		int type;
		int index;
		int value;
	};

	struct JoyDeviceMapping {

		String uid;
		Map<int,JoyEvent> buttons;
		Map<int,JoyEvent> axis;
		JoyEvent hat[HAT_MAX];
	};

	JoyEvent hat_map_default[HAT_MAX];

	Vector<JoyDeviceMapping> map_db;

	JoyEvent _find_to_event(String p_to);
	uint32_t _button_event(uint32_t p_last_id, int p_device, int p_index, bool p_pressed);
	uint32_t _axis_event(uint32_t p_last_id, int p_device, int p_axis, float p_value);

protected:


public:

	void joy_connection_changed(int p_idx, bool p_connected, String p_name, String p_guid="");

	void parse_mapping(String p_mapping);

	uint32_t joy_button(uint32_t p_last_id, int p_device, int p_button, bool p_pressed);
	uint32_t joy_axis(uint32_t p_last_id, int p_device, int p_axis, float p_value);
	uint32_t joy_hat(uint32_t p_last_id, int p_device, int p_val);

	InputPC();
	~InputPC();
};


#endif // INPUT_H
