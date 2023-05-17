/**************************************************************************/
/*  input_event.h                                                         */
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

#ifndef INPUT_EVENT_H
#define INPUT_EVENT_H

#include "core/math/transform_2d.h"
#include "core/resource.h"
#include "core/typedefs.h"
#include "core/ustring.h"

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
	BUTTON_XBUTTON1 = 8,
	BUTTON_XBUTTON2 = 9,
	BUTTON_MASK_LEFT = (1 << (BUTTON_LEFT - 1)),
	BUTTON_MASK_RIGHT = (1 << (BUTTON_RIGHT - 1)),
	BUTTON_MASK_MIDDLE = (1 << (BUTTON_MIDDLE - 1)),
	BUTTON_MASK_XBUTTON1 = (1 << (BUTTON_XBUTTON1 - 1)),
	BUTTON_MASK_XBUTTON2 = (1 << (BUTTON_XBUTTON2 - 1))
};

enum JoystickList {

	JOY_INVALID_OPTION = -1,

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
	JOY_BUTTON_16 = 16,
	JOY_BUTTON_17 = 17,
	JOY_BUTTON_18 = 18,
	JOY_BUTTON_19 = 19,
	JOY_BUTTON_20 = 20,
	JOY_BUTTON_21 = 21,
	JOY_BUTTON_22 = 22,
	JOY_BUTTON_MAX = 128, // Android supports up to 36 buttons. DirectInput supports up to 128 buttons.

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
	JOY_GUIDE = JOY_BUTTON_16,
	JOY_MISC1 = JOY_BUTTON_17,
	JOY_PADDLE1 = JOY_BUTTON_18,
	JOY_PADDLE2 = JOY_BUTTON_19,
	JOY_PADDLE3 = JOY_BUTTON_20,
	JOY_PADDLE4 = JOY_BUTTON_21,
	JOY_TOUCHPAD = JOY_BUTTON_22,

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

	JOY_WII_MINUS = JOY_BUTTON_10,
	JOY_WII_PLUS = JOY_BUTTON_11,

	JOY_VR_GRIP = JOY_BUTTON_2,
	JOY_VR_PAD = JOY_BUTTON_14,
	JOY_VR_TRIGGER = JOY_BUTTON_15,

	JOY_OCULUS_AX = JOY_BUTTON_7,
	JOY_OCULUS_BY = JOY_BUTTON_1,
	JOY_OCULUS_MENU = JOY_BUTTON_3,

	JOY_OPENVR_MENU = JOY_BUTTON_1,

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

	JOY_VR_ANALOG_TRIGGER = JOY_AXIS_2,
	JOY_VR_ANALOG_GRIP = JOY_AXIS_4,

	JOY_OPENVR_TOUCHPADX = JOY_AXIS_0,
	JOY_OPENVR_TOUCHPADY = JOY_AXIS_1,
};

enum MidiMessageList {
	MIDI_MESSAGE_NOTE_OFF = 0x8,
	MIDI_MESSAGE_NOTE_ON = 0x9,
	MIDI_MESSAGE_AFTERTOUCH = 0xA,
	MIDI_MESSAGE_CONTROL_CHANGE = 0xB,
	MIDI_MESSAGE_PROGRAM_CHANGE = 0xC,
	MIDI_MESSAGE_CHANNEL_PRESSURE = 0xD,
	MIDI_MESSAGE_PITCH_BEND = 0xE,
	MIDI_MESSAGE_SYSTEM_EXCLUSIVE = 0xF0,
	MIDI_MESSAGE_QUARTER_FRAME = 0xF1,
	MIDI_MESSAGE_SONG_POSITION_POINTER = 0xF2,
	MIDI_MESSAGE_SONG_SELECT = 0xF3,
	MIDI_MESSAGE_TUNE_REQUEST = 0xF6,
	MIDI_MESSAGE_TIMING_CLOCK = 0xF8,
	MIDI_MESSAGE_START = 0xFA,
	MIDI_MESSAGE_CONTINUE = 0xFB,
	MIDI_MESSAGE_STOP = 0xFC,
	MIDI_MESSAGE_ACTIVE_SENSING = 0xFE,
	MIDI_MESSAGE_SYSTEM_RESET = 0xFF,
};

/**
 * Input Modifier Status
 * for keyboard/mouse events.
 */

class InputEvent : public Resource {
	GDCLASS(InputEvent, Resource);

	int device;

protected:
	bool canceled = false;
	bool pressed = false;

	static void _bind_methods();

public:
	static const int DEVICE_ID_TOUCH_MOUSE;
	static const int DEVICE_ID_INTERNAL;

	void set_device(int p_device);
	int get_device() const;

	bool is_action(const StringName &p_action, bool p_exact_match = false) const;
	bool is_action_pressed(const StringName &p_action, bool p_allow_echo = false, bool p_exact_match = false) const;
	bool is_action_released(const StringName &p_action, bool p_exact_match = false) const;
	float get_action_strength(const StringName &p_action, bool p_exact_match = false) const;
	float get_action_raw_strength(const StringName &p_action, bool p_exact_match = false) const;

	bool is_canceled() const;
	bool is_pressed() const;
	bool is_released() const;
	virtual bool is_echo() const;
	// ...-.

	virtual String as_text() const;

	virtual Ref<InputEvent> xformed_by(const Transform2D &p_xform, const Vector2 &p_local_ofs = Vector2()) const;

	virtual bool action_match(const Ref<InputEvent> &p_event, bool p_exact_match, bool *p_pressed, float *p_strength, float *p_raw_strength, float p_deadzone) const;
	virtual bool shortcut_match(const Ref<InputEvent> &p_event, bool p_exact_match = true) const;
	virtual bool is_action_type() const;

	virtual bool accumulate(const Ref<InputEvent> &p_event) { return false; }
	InputEvent();
};

class InputEventWithModifiers : public InputEvent {
	GDCLASS(InputEventWithModifiers, InputEvent);

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

	uint32_t get_modifiers_mask() const;

	InputEventWithModifiers();
};

class InputEventKey : public InputEventWithModifiers {
	GDCLASS(InputEventKey, InputEventWithModifiers);

	uint32_t scancode; ///< check keyboard.h , KeyCode enum, without modifier masks
	uint32_t physical_scancode;
	uint32_t unicode; ///unicode

	bool echo; /// true if this is an echo key

protected:
	static void _bind_methods();

public:
	void set_pressed(bool p_pressed);

	void set_scancode(uint32_t p_scancode);
	uint32_t get_scancode() const;

	void set_physical_scancode(uint32_t p_scancode);
	uint32_t get_physical_scancode() const;

	void set_unicode(uint32_t p_unicode);
	uint32_t get_unicode() const;

	void set_echo(bool p_enable);
	virtual bool is_echo() const;

	uint32_t get_scancode_with_modifiers() const;
	uint32_t get_physical_scancode_with_modifiers() const;

	virtual bool action_match(const Ref<InputEvent> &p_event, bool p_exact_match, bool *p_pressed, float *p_strength, float *p_raw_strength, float p_deadzone) const;
	virtual bool shortcut_match(const Ref<InputEvent> &p_event, bool p_exact_match = true) const;

	virtual bool is_action_type() const { return true; }

	virtual String as_text() const;

	InputEventKey();
};

class InputEventMouse : public InputEventWithModifiers {
	GDCLASS(InputEventMouse, InputEventWithModifiers);

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
	GDCLASS(InputEventMouseButton, InputEventMouse);

	float factor;
	int button_index;
	bool doubleclick; //last even less than doubleclick time

protected:
	static void _bind_methods();

public:
	void set_factor(float p_factor);
	float get_factor() const;

	void set_button_index(int p_index);
	int get_button_index() const;

	void set_pressed(bool p_pressed);
	void set_canceled(bool p_canceled);

	void set_doubleclick(bool p_doubleclick);
	bool is_doubleclick() const;

	virtual Ref<InputEvent> xformed_by(const Transform2D &p_xform, const Vector2 &p_local_ofs = Vector2()) const;
	virtual bool action_match(const Ref<InputEvent> &p_event, bool p_exact_match, bool *p_pressed, float *p_strength, float *p_raw_strength, float p_deadzone) const;
	virtual bool shortcut_match(const Ref<InputEvent> &p_event, bool p_exact_match = true) const;

	virtual bool is_action_type() const { return true; }
	virtual String as_text() const;

	InputEventMouseButton();
};

class InputEventMouseMotion : public InputEventMouse {
	GDCLASS(InputEventMouseMotion, InputEventMouse);

	Vector2 tilt;
	float pressure;
	Vector2 relative;
	Vector2 speed;
	bool pen_inverted;

protected:
	static void _bind_methods();

public:
	void set_tilt(const Vector2 &p_tilt);
	Vector2 get_tilt() const;

	void set_pressure(float p_pressure);
	float get_pressure() const;

	void set_pen_inverted(bool p_inverted);
	bool get_pen_inverted() const;

	void set_relative(const Vector2 &p_relative);
	Vector2 get_relative() const;

	void set_speed(const Vector2 &p_speed);
	Vector2 get_speed() const;

	virtual Ref<InputEvent> xformed_by(const Transform2D &p_xform, const Vector2 &p_local_ofs = Vector2()) const;
	virtual String as_text() const;

	virtual bool accumulate(const Ref<InputEvent> &p_event);

	InputEventMouseMotion();
};

class InputEventJoypadMotion : public InputEvent {
	GDCLASS(InputEventJoypadMotion, InputEvent);
	int axis; ///< Joypad axis
	float axis_value; ///< -1 to 1

protected:
	static void _bind_methods();

public:
	void set_axis(int p_axis);
	int get_axis() const;

	void set_axis_value(float p_value);
	float get_axis_value() const;

	virtual bool action_match(const Ref<InputEvent> &p_event, bool p_exact_match, bool *p_pressed, float *p_strength, float *p_raw_strength, float p_deadzone) const;
	virtual bool shortcut_match(const Ref<InputEvent> &p_event, bool p_exact_match = true) const;

	virtual bool is_action_type() const { return true; }
	virtual String as_text() const;

	InputEventJoypadMotion();
};

class InputEventJoypadButton : public InputEvent {
	GDCLASS(InputEventJoypadButton, InputEvent);

	int button_index;
	float pressure; //0 to 1
protected:
	static void _bind_methods();

public:
	void set_button_index(int p_index);
	int get_button_index() const;

	void set_pressed(bool p_pressed);

	void set_pressure(float p_pressure);
	float get_pressure() const;

	virtual bool action_match(const Ref<InputEvent> &p_event, bool p_exact_match, bool *p_pressed, float *p_strength, float *p_raw_strength, float p_deadzone) const;
	virtual bool shortcut_match(const Ref<InputEvent> &p_event, bool p_exact_match = true) const;

	virtual bool is_action_type() const { return true; }
	virtual String as_text() const;

	InputEventJoypadButton();
};

class InputEventScreenTouch : public InputEvent {
	GDCLASS(InputEventScreenTouch, InputEvent);
	int index;
	Vector2 pos;
	bool double_tap;

protected:
	static void _bind_methods();

public:
	void set_index(int p_index);
	int get_index() const;

	void set_position(const Vector2 &p_pos);
	Vector2 get_position() const;

	void set_pressed(bool p_pressed);
	void set_canceled(bool p_canceled);

	void set_double_tap(bool p_double_tap);
	bool is_double_tap() const;

	virtual Ref<InputEvent> xformed_by(const Transform2D &p_xform, const Vector2 &p_local_ofs = Vector2()) const;
	virtual String as_text() const;

	InputEventScreenTouch();
};

class InputEventScreenDrag : public InputEvent {
	GDCLASS(InputEventScreenDrag, InputEvent);
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

	virtual bool accumulate(const Ref<InputEvent> &p_event);

	InputEventScreenDrag();
};

class InputEventAction : public InputEvent {
	GDCLASS(InputEventAction, InputEvent);

	StringName action;
	float strength;

protected:
	static void _bind_methods();

public:
	void set_action(const StringName &p_action);
	StringName get_action() const;

	void set_pressed(bool p_pressed);

	void set_strength(float p_strength);
	float get_strength() const;

	virtual bool is_action(const StringName &p_action) const;

	virtual bool action_match(const Ref<InputEvent> &p_event, bool p_exact_match, bool *p_pressed, float *p_strength, float *p_raw_strength, float p_deadzone) const;
	virtual bool shortcut_match(const Ref<InputEvent> &p_event, bool p_exact_match = true) const;

	virtual bool is_action_type() const { return true; }
	virtual String as_text() const;

	InputEventAction();
};

class InputEventGesture : public InputEventWithModifiers {
	GDCLASS(InputEventGesture, InputEventWithModifiers);

	Vector2 pos;

protected:
	static void _bind_methods();

public:
	void set_position(const Vector2 &p_pos);
	Vector2 get_position() const;
};

class InputEventMagnifyGesture : public InputEventGesture {
	GDCLASS(InputEventMagnifyGesture, InputEventGesture);
	real_t factor;

protected:
	static void _bind_methods();

public:
	void set_factor(real_t p_factor);
	real_t get_factor() const;

	virtual Ref<InputEvent> xformed_by(const Transform2D &p_xform, const Vector2 &p_local_ofs = Vector2()) const;
	virtual String as_text() const;

	InputEventMagnifyGesture();
};

class InputEventPanGesture : public InputEventGesture {
	GDCLASS(InputEventPanGesture, InputEventGesture);
	Vector2 delta;

protected:
	static void _bind_methods();

public:
	void set_delta(const Vector2 &p_delta);
	Vector2 get_delta() const;

	virtual Ref<InputEvent> xformed_by(const Transform2D &p_xform, const Vector2 &p_local_ofs = Vector2()) const;
	virtual String as_text() const;

	InputEventPanGesture();
};

class InputEventMIDI : public InputEvent {
	GDCLASS(InputEventMIDI, InputEvent);

	int channel;
	int message;
	int pitch;
	int velocity;
	int instrument;
	int pressure;
	int controller_number;
	int controller_value;

protected:
	static void _bind_methods();

public:
	void set_channel(const int p_channel);
	int get_channel() const;

	void set_message(const int p_message);
	int get_message() const;

	void set_pitch(const int p_pitch);
	int get_pitch() const;

	void set_velocity(const int p_velocity);
	int get_velocity() const;

	void set_instrument(const int p_instrument);
	int get_instrument() const;

	void set_pressure(const int p_pressure);
	int get_pressure() const;

	void set_controller_number(const int p_controller_number);
	int get_controller_number() const;

	void set_controller_value(const int p_controller_value);
	int get_controller_value() const;

	virtual String as_text() const;

	InputEventMIDI();
};

#endif // INPUT_EVENT_H
