/*************************************************************************/
/*  input_event.h                                                        */
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

#ifndef INPUT_EVENT_H
#define INPUT_EVENT_H

#include "core/math/transform_2d.h"
#include "core/os/copymem.h"
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

enum JoyButtonList {

	JOY_INVALID_BUTTON = -1,

	// SDL Buttons
	JOY_BUTTON_A = 0,
	JOY_BUTTON_B = 1,
	JOY_BUTTON_X = 2,
	JOY_BUTTON_Y = 3,
	JOY_BUTTON_BACK = 4,
	JOY_BUTTON_GUIDE = 5,
	JOY_BUTTON_START = 6,
	JOY_BUTTON_LEFT_STICK = 7,
	JOY_BUTTON_RIGHT_STICK = 8,
	JOY_BUTTON_LEFT_SHOULDER = 9,
	JOY_BUTTON_RIGHT_SHOULDER = 10,
	JOY_BUTTON_DPAD_UP = 11,
	JOY_BUTTON_DPAD_DOWN = 12,
	JOY_BUTTON_DPAD_LEFT = 13,
	JOY_BUTTON_DPAD_RIGHT = 14,
	JOY_SDL_BUTTONS = 15,

	// Sony Buttons
	JOY_SONY_X = JOY_BUTTON_A,
	JOY_SONY_CROSS = JOY_BUTTON_A,
	JOY_SONY_CIRCLE = JOY_BUTTON_B,
	JOY_SONY_SQUARE = JOY_BUTTON_X,
	JOY_SONY_TRIANGLE = JOY_BUTTON_Y,
	JOY_SONY_SELECT = JOY_BUTTON_BACK,
	JOY_SONY_START = JOY_BUTTON_START,
	JOY_SONY_PS = JOY_BUTTON_GUIDE,
	JOY_SONY_L1 = JOY_BUTTON_LEFT_SHOULDER,
	JOY_SONY_R1 = JOY_BUTTON_RIGHT_SHOULDER,
	JOY_SONY_L3 = JOY_BUTTON_LEFT_STICK,
	JOY_SONY_R3 = JOY_BUTTON_RIGHT_STICK,

	// Xbox Buttons
	JOY_XBOX_A = JOY_BUTTON_A,
	JOY_XBOX_B = JOY_BUTTON_B,
	JOY_XBOX_X = JOY_BUTTON_X,
	JOY_XBOX_Y = JOY_BUTTON_Y,
	JOY_XBOX_BACK = JOY_BUTTON_BACK,
	JOY_XBOX_START = JOY_BUTTON_START,
	JOY_XBOX_HOME = JOY_BUTTON_GUIDE,
	JOY_XBOX_LS = JOY_BUTTON_LEFT_STICK,
	JOY_XBOX_RS = JOY_BUTTON_RIGHT_STICK,
	JOY_XBOX_LB = JOY_BUTTON_LEFT_SHOULDER,
	JOY_XBOX_RB = JOY_BUTTON_RIGHT_SHOULDER,

	JOY_BUTTON_MAX = 36 // Apparently Android supports up to 36 buttons.
};

enum JoyAxisList {

	JOY_INVALID_AXIS = -1,

	// SDL Axes
	JOY_AXIS_LEFT_X = 0,
	JOY_AXIS_LEFT_Y = 1,
	JOY_AXIS_RIGHT_X = 2,
	JOY_AXIS_RIGHT_Y = 3,
	JOY_AXIS_TRIGGER_LEFT = 4,
	JOY_AXIS_TRIGGER_RIGHT = 5,
	JOY_SDL_AXES = 6,

	// Joystick axes.
	JOY_AXIS_0_X = 0,
	JOY_AXIS_0_Y = 1,
	JOY_AXIS_1_X = 2,
	JOY_AXIS_1_Y = 3,
	JOY_AXIS_2_X = 4,
	JOY_AXIS_2_Y = 5,
	JOY_AXIS_3_X = 6,
	JOY_AXIS_3_Y = 7,
	JOY_AXIS_4_X = 8,
	JOY_AXIS_4_Y = 9,

	JOY_AXIS_MAX = 10 // OpenVR supports up to 5 Joysticks making a total of 10 axes.
};

enum MidiMessageList {
	MIDI_MESSAGE_NOTE_OFF = 0x8,
	MIDI_MESSAGE_NOTE_ON = 0x9,
	MIDI_MESSAGE_AFTERTOUCH = 0xA,
	MIDI_MESSAGE_CONTROL_CHANGE = 0xB,
	MIDI_MESSAGE_PROGRAM_CHANGE = 0xC,
	MIDI_MESSAGE_CHANNEL_PRESSURE = 0xD,
	MIDI_MESSAGE_PITCH_BEND = 0xE,
};

/**
 * Input Modifier Status
 * for keyboard/mouse events.
 */

class InputEvent : public Resource {
	GDCLASS(InputEvent, Resource);

	int device = 0;

protected:
	static void _bind_methods();

public:
	static const int DEVICE_ID_TOUCH_MOUSE;
	static const int DEVICE_ID_INTERNAL;

	void set_device(int p_device);
	int get_device() const;

	bool is_action(const StringName &p_action) const;
	bool is_action_pressed(const StringName &p_action, bool p_allow_echo = false) const;
	bool is_action_released(const StringName &p_action) const;
	float get_action_strength(const StringName &p_action) const;

	// To be removed someday, since they do not make sense for all events
	virtual bool is_pressed() const;
	virtual bool is_echo() const;

	virtual String as_text() const;

	virtual Ref<InputEvent> xformed_by(const Transform2D &p_xform, const Vector2 &p_local_ofs = Vector2()) const;

	virtual bool action_match(const Ref<InputEvent> &p_event, bool *p_pressed, float *p_strength, float p_deadzone) const;
	virtual bool shortcut_match(const Ref<InputEvent> &p_event) const;
	virtual bool is_action_type() const;

	virtual bool accumulate(const Ref<InputEvent> &p_event) { return false; }

	InputEvent() {}
};

class InputEventFromWindow : public InputEvent {
	GDCLASS(InputEventFromWindow, InputEvent);

	int64_t window_id = 0;

protected:
	static void _bind_methods();

public:
	void set_window_id(int64_t p_id);
	int64_t get_window_id() const;

	InputEventFromWindow() {}
};

class InputEventWithModifiers : public InputEventFromWindow {
	GDCLASS(InputEventWithModifiers, InputEventFromWindow);

	bool shift = false;
	bool alt = false;
#ifdef APPLE_STYLE_KEYS
	union {
		bool command;
		bool meta = false; //< windows/mac key
	};

	bool control = false;
#else
	union {
		bool command; //< windows/mac key
		bool control = false;
	};
	bool meta = false; //< windows/mac key
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

	InputEventWithModifiers() {}
};

class InputEventKey : public InputEventWithModifiers {
	GDCLASS(InputEventKey, InputEventWithModifiers);

	bool pressed = false; /// otherwise release

	uint32_t keycode = 0; ///< check keyboard.h , KeyCode enum, without modifier masks
	uint32_t physical_keycode = 0;
	uint32_t unicode = 0; ///unicode

	bool echo = false; /// true if this is an echo key

protected:
	static void _bind_methods();

public:
	void set_pressed(bool p_pressed);
	virtual bool is_pressed() const override;

	void set_keycode(uint32_t p_keycode);
	uint32_t get_keycode() const;

	void set_physical_keycode(uint32_t p_keycode);
	uint32_t get_physical_keycode() const;

	void set_unicode(uint32_t p_unicode);
	uint32_t get_unicode() const;

	void set_echo(bool p_enable);
	virtual bool is_echo() const override;

	uint32_t get_keycode_with_modifiers() const;
	uint32_t get_physical_keycode_with_modifiers() const;

	virtual bool action_match(const Ref<InputEvent> &p_event, bool *p_pressed, float *p_strength, float p_deadzone) const override;
	virtual bool shortcut_match(const Ref<InputEvent> &p_event) const override;

	virtual bool is_action_type() const override { return true; }

	virtual String as_text() const override;

	InputEventKey() {}
};

class InputEventMouse : public InputEventWithModifiers {
	GDCLASS(InputEventMouse, InputEventWithModifiers);

	int button_mask = 0;

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

	InputEventMouse() {}
};

class InputEventMouseButton : public InputEventMouse {
	GDCLASS(InputEventMouseButton, InputEventMouse);

	float factor = 1;
	int button_index = 0;
	bool pressed = false; //otherwise released
	bool doubleclick = false; //last even less than doubleclick time

protected:
	static void _bind_methods();

public:
	void set_factor(float p_factor);
	float get_factor() const;

	void set_button_index(int p_index);
	int get_button_index() const;

	void set_pressed(bool p_pressed);
	virtual bool is_pressed() const override;

	void set_doubleclick(bool p_doubleclick);
	bool is_doubleclick() const;

	virtual Ref<InputEvent> xformed_by(const Transform2D &p_xform, const Vector2 &p_local_ofs = Vector2()) const override;
	virtual bool action_match(const Ref<InputEvent> &p_event, bool *p_pressed, float *p_strength, float p_deadzone) const override;

	virtual bool is_action_type() const override { return true; }
	virtual String as_text() const override;

	InputEventMouseButton() {}
};

class InputEventMouseMotion : public InputEventMouse {
	GDCLASS(InputEventMouseMotion, InputEventMouse);

	Vector2 tilt;
	float pressure = 0;
	Vector2 relative;
	Vector2 speed;

protected:
	static void _bind_methods();

public:
	void set_tilt(const Vector2 &p_tilt);
	Vector2 get_tilt() const;

	void set_pressure(float p_pressure);
	float get_pressure() const;

	void set_relative(const Vector2 &p_relative);
	Vector2 get_relative() const;

	void set_speed(const Vector2 &p_speed);
	Vector2 get_speed() const;

	virtual Ref<InputEvent> xformed_by(const Transform2D &p_xform, const Vector2 &p_local_ofs = Vector2()) const override;
	virtual String as_text() const override;

	virtual bool accumulate(const Ref<InputEvent> &p_event) override;

	InputEventMouseMotion() {}
};

class InputEventJoypadMotion : public InputEvent {
	GDCLASS(InputEventJoypadMotion, InputEvent);
	int axis = 0; ///< Joypad axis
	float axis_value = 0; ///< -1 to 1

protected:
	static void _bind_methods();

public:
	void set_axis(int p_axis);
	int get_axis() const;

	void set_axis_value(float p_value);
	float get_axis_value() const;

	virtual bool is_pressed() const override;

	virtual bool action_match(const Ref<InputEvent> &p_event, bool *p_pressed, float *p_strength, float p_deadzone) const override;

	virtual bool is_action_type() const override { return true; }
	virtual String as_text() const override;

	InputEventJoypadMotion() {}
};

class InputEventJoypadButton : public InputEvent {
	GDCLASS(InputEventJoypadButton, InputEvent);

	int button_index = 0;
	bool pressed = false;
	float pressure = 0; //0 to 1
protected:
	static void _bind_methods();

public:
	void set_button_index(int p_index);
	int get_button_index() const;

	void set_pressed(bool p_pressed);
	virtual bool is_pressed() const override;

	void set_pressure(float p_pressure);
	float get_pressure() const;

	virtual bool action_match(const Ref<InputEvent> &p_event, bool *p_pressed, float *p_strength, float p_deadzone) const override;
	virtual bool shortcut_match(const Ref<InputEvent> &p_event) const override;

	virtual bool is_action_type() const override { return true; }
	virtual String as_text() const override;

	InputEventJoypadButton() {}
};

class InputEventScreenTouch : public InputEventFromWindow {
	GDCLASS(InputEventScreenTouch, InputEventFromWindow);
	int index = 0;
	Vector2 pos;
	bool pressed = false;

protected:
	static void _bind_methods();

public:
	void set_index(int p_index);
	int get_index() const;

	void set_position(const Vector2 &p_pos);
	Vector2 get_position() const;

	void set_pressed(bool p_pressed);
	virtual bool is_pressed() const override;

	virtual Ref<InputEvent> xformed_by(const Transform2D &p_xform, const Vector2 &p_local_ofs = Vector2()) const override;
	virtual String as_text() const override;

	InputEventScreenTouch() {}
};

class InputEventScreenDrag : public InputEventFromWindow {
	GDCLASS(InputEventScreenDrag, InputEventFromWindow);
	int index = 0;
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

	virtual Ref<InputEvent> xformed_by(const Transform2D &p_xform, const Vector2 &p_local_ofs = Vector2()) const override;
	virtual String as_text() const override;

	InputEventScreenDrag() {}
};

class InputEventAction : public InputEvent {
	GDCLASS(InputEventAction, InputEvent);

	StringName action;
	bool pressed = false;
	float strength = 1.0f;

protected:
	static void _bind_methods();

public:
	void set_action(const StringName &p_action);
	StringName get_action() const;

	void set_pressed(bool p_pressed);
	virtual bool is_pressed() const override;

	void set_strength(float p_strength);
	float get_strength() const;

	virtual bool is_action(const StringName &p_action) const;

	virtual bool action_match(const Ref<InputEvent> &p_event, bool *p_pressed, float *p_strength, float p_deadzone) const override;

	virtual bool shortcut_match(const Ref<InputEvent> &p_event) const override;
	virtual bool is_action_type() const override { return true; }
	virtual String as_text() const override;

	InputEventAction() {}
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
	real_t factor = 1.0;

protected:
	static void _bind_methods();

public:
	void set_factor(real_t p_factor);
	real_t get_factor() const;

	virtual Ref<InputEvent> xformed_by(const Transform2D &p_xform, const Vector2 &p_local_ofs = Vector2()) const override;
	virtual String as_text() const override;

	InputEventMagnifyGesture() {}
};

class InputEventPanGesture : public InputEventGesture {
	GDCLASS(InputEventPanGesture, InputEventGesture);
	Vector2 delta;

protected:
	static void _bind_methods();

public:
	void set_delta(const Vector2 &p_delta);
	Vector2 get_delta() const;

	virtual Ref<InputEvent> xformed_by(const Transform2D &p_xform, const Vector2 &p_local_ofs = Vector2()) const override;
	virtual String as_text() const override;

	InputEventPanGesture() {}
};

class InputEventMIDI : public InputEvent {
	GDCLASS(InputEventMIDI, InputEvent);

	int channel = 0;
	int message = 0;
	int pitch = 0;
	int velocity = 0;
	int instrument = 0;
	int pressure = 0;
	int controller_number = 0;
	int controller_value = 0;

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

	virtual String as_text() const override;

	InputEventMIDI() {}
};

#endif // INPUT_EVENT_H
