/*************************************************************************/
/*  input_event.h                                                        */
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
#ifndef INPUT_EVENT_H
#define INPUT_EVENT_H

#include "math_2d.h"
#include "os/copymem.h"
#include "resource.h"
#include "typedefs.h"
#include "ustring.h"
/**
	@author Juan Linietsky <reduzio@gmail.com>
*/

/**
 * Input Event classes. These are used in the main loop.
 * The events are pretty obvious.
 */

enum ButtonList {
	BUTTON_LEFT = 1,
	BUTTON_RIGHT = 2,
	BUTTON_MIDDLE = 3,
	BUTTON_WHEEL_UP = 4,
	BUTTON_WHEEL_DOWN = 5,
	BUTTON_WHEEL_LEFT = 6,
	BUTTON_WHEEL_RIGHT = 7,
	BUTTON_MASK_LEFT = (1 << (BUTTON_LEFT - 1)),
	BUTTON_MASK_RIGHT = (1 << (BUTTON_RIGHT - 1)),
	BUTTON_MASK_MIDDLE = (1 << (BUTTON_MIDDLE - 1)),

};

enum JoystickList {

	JOY_BUTTON_0 = 0,
	JOY_BUTTON_1 = 1,
	JOY_BUTTON_2 = 2,
	JOY_BUTTON_3 = 3,
	JOY_BUTTON_4 = 4,
	JOY_BUTTON_5 = 5,
	JOY_BUTTON_6 = 6,
	JOY_BUTTON_7 = 7,
	JOY_BUTTON_8 = 8,
	JOY_BUTTON_9 = 9,
	JOY_BUTTON_10 = 10,
	JOY_BUTTON_11 = 11,
	JOY_BUTTON_12 = 12,
	JOY_BUTTON_13 = 13,
	JOY_BUTTON_14 = 14,
	JOY_BUTTON_15 = 15,
	JOY_BUTTON_MAX = 16,

	JOY_L = JOY_BUTTON_4,
	JOY_R = JOY_BUTTON_5,
	JOY_L2 = JOY_BUTTON_6,
	JOY_R2 = JOY_BUTTON_7,
	JOY_L3 = JOY_BUTTON_8,
	JOY_R3 = JOY_BUTTON_9,
	JOY_SELECT = JOY_BUTTON_10,
	JOY_START = JOY_BUTTON_11,
	JOY_DPAD_UP = JOY_BUTTON_12,
	JOY_DPAD_DOWN = JOY_BUTTON_13,
	JOY_DPAD_LEFT = JOY_BUTTON_14,
	JOY_DPAD_RIGHT = JOY_BUTTON_15,

	JOY_SONY_CIRCLE = JOY_BUTTON_1,
	JOY_SONY_X = JOY_BUTTON_0,
	JOY_SONY_SQUARE = JOY_BUTTON_2,
	JOY_SONY_TRIANGLE = JOY_BUTTON_3,

	JOY_XBOX_A = JOY_BUTTON_0,
	JOY_XBOX_B = JOY_BUTTON_1,
	JOY_XBOX_X = JOY_BUTTON_2,
	JOY_XBOX_Y = JOY_BUTTON_3,

	JOY_DS_A = JOY_BUTTON_1,
	JOY_DS_B = JOY_BUTTON_0,
	JOY_DS_X = JOY_BUTTON_3,
	JOY_DS_Y = JOY_BUTTON_2,

	JOY_WII_C = JOY_BUTTON_5,
	JOY_WII_Z = JOY_BUTTON_6,

	JOY_WII_MINUS = JOY_BUTTON_9,
	JOY_WII_PLUS = JOY_BUTTON_10,

	// end of history

	JOY_AXIS_0 = 0,
	JOY_AXIS_1 = 1,
	JOY_AXIS_2 = 2,
	JOY_AXIS_3 = 3,
	JOY_AXIS_4 = 4,
	JOY_AXIS_5 = 5,
	JOY_AXIS_6 = 6,
	JOY_AXIS_7 = 7,
	JOY_AXIS_8 = 8,
	JOY_AXIS_9 = 9,
	JOY_AXIS_MAX = 10,

	JOY_ANALOG_LX = JOY_AXIS_0,
	JOY_ANALOG_LY = JOY_AXIS_1,

	JOY_ANALOG_RX = JOY_AXIS_2,
	JOY_ANALOG_RY = JOY_AXIS_3,

	JOY_ANALOG_L2 = JOY_AXIS_6,
	JOY_ANALOG_R2 = JOY_AXIS_7,
};

/**
 * Input Modifier Status
 * for keyboard/mouse events.
 */

class InputEvent : public Resource {
	GDCLASS(InputEvent, Resource)

	uint32_t id;
	int device;

protected:
	static void _bind_methods();

public:
	void set_id(uint32_t p_id);
	uint32_t get_id() const;

	void set_device(int p_device);
	int get_device() const;

	virtual bool is_pressed() const;
	virtual bool is_action(const StringName &p_action) const;
	virtual bool is_action_pressed(const StringName &p_action) const;
	virtual bool is_action_released(const StringName &p_action) const;
	virtual bool is_echo() const;
	virtual String as_text() const;

	virtual Ref<InputEvent> xformed_by(const Transform2D &p_xform, const Vector2 &p_local_ofs = Vector2()) const;

	virtual bool action_match(const Ref<InputEvent> &p_event) const;
	virtual bool shortcut_match(const Ref<InputEvent> &p_event) const;
	virtual bool is_action_type() const;

	InputEvent();
};

class InputEventWithModifiers : public InputEvent {
	GDCLASS(InputEventWithModifiers, InputEvent)

	bool shift;
	bool alt;
#ifdef APPLE_STYLE_KEYS
	union {
		bool command;
		bool meta; //< windows/mac key
	};

	bool control;
#else
	union {
		bool command; //< windows/mac key
		bool control;
	};
	bool meta; //< windows/mac key

#endif

protected:
	static void _bind_methods();

public:
	void set_shift(bool p_enabled);
	bool get_shift() const;

	void set_alt(bool p_enabled);
	bool get_alt() const;

	void set_control(bool p_enabled);
	bool get_control() const;

	void set_metakey(bool p_enabled);
	bool get_metakey() const;

	void set_command(bool p_enabled);
	bool get_command() const;

	void set_modifiers_from_event(const InputEventWithModifiers *event);

	InputEventWithModifiers();
};

class InputEventKey : public InputEventWithModifiers {

	GDCLASS(InputEventKey, InputEventWithModifiers)

	bool pressed; /// otherwise release

	uint32_t scancode; ///< check keyboard.h , KeyCode enum, without modifier masks
	uint32_t unicode; ///unicode

	bool echo; /// true if this is an echo key

protected:
	static void _bind_methods();

public:
	void set_pressed(bool p_pressed);
	virtual bool is_pressed() const;

	void set_scancode(uint32_t p_scancode);
	uint32_t get_scancode() const;

	void set_unicode(uint32_t p_unicode);
	uint32_t get_unicode() const;

	void set_echo(bool p_enable);
	virtual bool is_echo() const;

	uint32_t get_scancode_with_modifiers() const;

	virtual bool action_match(const Ref<InputEvent> &p_event) const;
	virtual bool shortcut_match(const Ref<InputEvent> &p_event) const;

	virtual bool is_action_type() const { return true; }

	virtual String as_text() const;

	InputEventKey();
};

class InputEventMouse : public InputEventWithModifiers {

	GDCLASS(InputEventMouse, InputEventWithModifiers)

	int button_mask;

	Vector2 pos;
	Vector2 global_pos;

protected:
	static void _bind_methods();

public:
	void set_button_mask(int p_mask);
	int get_button_mask() const;

	void set_position(const Vector2 &p_pos);
	Vector2 get_position() const;

	void set_global_position(const Vector2 &p_global_pos);
	Vector2 get_global_position() const;

	InputEventMouse();
};

class InputEventMouseButton : public InputEventMouse {

	GDCLASS(InputEventMouseButton, InputEventMouse)

	float factor;
	int button_index;
	bool pressed; //otherwise released
	bool doubleclick; //last even less than doubleclick time

protected:
	static void _bind_methods();

public:
	void set_factor(float p_factor);
	float get_factor();

	void set_button_index(int p_index);
	int get_button_index() const;

	void set_pressed(bool p_pressed);
	virtual bool is_pressed() const;

	void set_doubleclick(bool p_doubleclick);
	bool is_doubleclick() const;

	virtual Ref<InputEvent> xformed_by(const Transform2D &p_xform, const Vector2 &p_local_ofs = Vector2()) const;
	virtual bool action_match(const Ref<InputEvent> &p_event) const;

	virtual bool is_action_type() const { return true; }
	virtual String as_text() const;

	InputEventMouseButton();
};

class InputEventMouseMotion : public InputEventMouse {

	GDCLASS(InputEventMouseMotion, InputEventMouse)
	Vector2 relative;
	Vector2 speed;

protected:
	static void _bind_methods();

public:
	void set_relative(const Vector2 &p_relative);
	Vector2 get_relative() const;

	void set_speed(const Vector2 &p_speed);
	Vector2 get_speed() const;

	virtual Ref<InputEvent> xformed_by(const Transform2D &p_xform, const Vector2 &p_local_ofs = Vector2()) const;
	virtual String as_text() const;

	InputEventMouseMotion();
};

class InputEventJoypadMotion : public InputEvent {

	GDCLASS(InputEventJoypadMotion, InputEvent)
	int axis; ///< Joypad axis
	float axis_value; ///< -1 to 1

protected:
	static void _bind_methods();

public:
	void set_axis(int p_axis);
	int get_axis() const;

	void set_axis_value(float p_value);
	float get_axis_value() const;

	virtual bool is_pressed() const;
	virtual bool action_match(const Ref<InputEvent> &p_event) const;

	virtual bool is_action_type() const { return true; }
	virtual String as_text() const;

	InputEventJoypadMotion();
};

class InputEventJoypadButton : public InputEvent {
	GDCLASS(InputEventJoypadButton, InputEvent)

	int button_index;
	bool pressed;
	float pressure; //0 to 1
protected:
	static void _bind_methods();

public:
	void set_button_index(int p_index);
	int get_button_index() const;

	void set_pressed(bool p_pressed);
	virtual bool is_pressed() const;

	void set_pressure(float p_pressure);
	float get_pressure() const;

	virtual bool action_match(const Ref<InputEvent> &p_event) const;

	virtual bool is_action_type() const { return true; }
	virtual String as_text() const;

	InputEventJoypadButton();
};

class InputEventScreenTouch : public InputEvent {
	GDCLASS(InputEventScreenTouch, InputEvent)
	int index;
	Vector2 pos;
	bool pressed;

protected:
	static void _bind_methods();

public:
	void set_index(int p_index);
	int get_index() const;

	void set_position(const Vector2 &p_pos);
	Vector2 get_position() const;

	void set_pressed(bool p_pressed);
	virtual bool is_pressed() const;

	virtual Ref<InputEvent> xformed_by(const Transform2D &p_xform, const Vector2 &p_local_ofs = Vector2()) const;
	virtual String as_text() const;

	InputEventScreenTouch();
};

class InputEventScreenDrag : public InputEvent {

	GDCLASS(InputEventScreenDrag, InputEvent)
	int index;
	Vector2 pos;
	Vector2 relative;
	Vector2 speed;

protected:
	static void _bind_methods();

public:
	void set_index(int p_index);
	int get_index() const;

	void set_position(const Vector2 &p_pos);
	Vector2 get_position() const;

	void set_relative(const Vector2 &p_relative);
	Vector2 get_relative() const;

	void set_speed(const Vector2 &p_speed);
	Vector2 get_speed() const;

	virtual Ref<InputEvent> xformed_by(const Transform2D &p_xform, const Vector2 &p_local_ofs = Vector2()) const;
	virtual String as_text() const;

	InputEventScreenDrag();
};

class InputEventAction : public InputEvent {

	GDCLASS(InputEventAction, InputEvent)

	StringName action;
	bool pressed;

protected:
	static void _bind_methods();

public:
	void set_action(const StringName &p_action);
	StringName get_action() const;

	void set_pressed(bool p_pressed);
	virtual bool is_pressed() const;

	virtual bool is_action(const StringName &p_action) const;

	virtual bool is_action_type() const { return true; }
	virtual String as_text() const;

	InputEventAction();
};

class InputEventGesture : public InputEventWithModifiers {

	GDCLASS(InputEventGesture, InputEventWithModifiers)

	Vector2 pos;

protected:
	static void _bind_methods();

public:
	void set_position(const Vector2 &p_pos);
	Vector2 get_position() const;
};

class InputEventMagnifyGesture : public InputEventGesture {

	GDCLASS(InputEventMagnifyGesture, InputEventGesture)
	real_t factor;

protected:
	static void _bind_methods();

public:
	void set_factor(real_t p_factor);
	real_t get_factor() const;

	virtual Ref<InputEvent> xformed_by(const Transform2D &p_xform, const Vector2 &p_local_ofs = Vector2()) const;

	InputEventMagnifyGesture();
};

class InputEventPanGesture : public InputEventGesture {

	GDCLASS(InputEventPanGesture, InputEventGesture)
	Vector2 delta;

protected:
	static void _bind_methods();

public:
	void set_delta(const Vector2 &p_delta);
	Vector2 get_delta() const;

	virtual Ref<InputEvent> xformed_by(const Transform2D &p_xform, const Vector2 &p_local_ofs = Vector2()) const;

	InputEventPanGesture();
};
#endif
