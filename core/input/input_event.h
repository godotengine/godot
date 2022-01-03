/*************************************************************************/
/*  input_event.h                                                        */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "core/input/input_enums.h"
#include "core/io/resource.h"
#include "core/math/transform_2d.h"
#include "core/os/keyboard.h"
#include "core/string/ustring.h"
#include "core/typedefs.h"

/**
 * Input Event classes. These are used in the main loop.
 * The events are pretty obvious.
 */

class Shortcut;

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

	bool is_action(const StringName &p_action, bool p_exact_match = false) const;
	bool is_action_pressed(const StringName &p_action, bool p_allow_echo = false, bool p_exact_match = false) const;
	bool is_action_released(const StringName &p_action, bool p_exact_match = false) const;
	float get_action_strength(const StringName &p_action, bool p_exact_match = false) const;
	float get_action_raw_strength(const StringName &p_action, bool p_exact_match = false) const;

	// To be removed someday, since they do not make sense for all events
	virtual bool is_pressed() const;
	virtual bool is_echo() const;

	virtual String as_text() const = 0;

	virtual Ref<InputEvent> xformed_by(const Transform2D &p_xform, const Vector2 &p_local_ofs = Vector2()) const;

	virtual bool action_match(const Ref<InputEvent> &p_event, bool *p_pressed, float *p_strength, float *p_raw_strength, float p_deadzone) const;
	virtual bool is_match(const Ref<InputEvent> &p_event, bool p_exact_match = true) const;

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

	bool store_command = true;

	bool shift_pressed = false;
	bool alt_pressed = false;
#ifdef APPLE_STYLE_KEYS
	union {
		bool command_pressed;
		bool meta_pressed = false; //< windows/mac key
	};

	bool ctrl_pressed = false;
#else
	union {
		bool command_pressed; //< windows/mac key
		bool ctrl_pressed = false;
	};
	bool meta_pressed = false; //< windows/mac key
#endif

protected:
	static void _bind_methods();
	virtual void _validate_property(PropertyInfo &property) const override;

public:
	void set_store_command(bool p_enabled);
	bool is_storing_command() const;

	void set_shift_pressed(bool p_pressed);
	bool is_shift_pressed() const;

	void set_alt_pressed(bool p_pressed);
	bool is_alt_pressed() const;

	void set_ctrl_pressed(bool p_pressed);
	bool is_ctrl_pressed() const;

	void set_meta_pressed(bool p_pressed);
	bool is_meta_pressed() const;

	void set_command_pressed(bool p_pressed);
	bool is_command_pressed() const;

	void set_modifiers_from_event(const InputEventWithModifiers *event);

	Key get_modifiers_mask() const;

	virtual String as_text() const override;
	virtual String to_string() override;

	InputEventWithModifiers() {}
};

class InputEventKey : public InputEventWithModifiers {
	GDCLASS(InputEventKey, InputEventWithModifiers);

	bool pressed = false; /// otherwise release

	Key keycode = Key::NONE; // Key enum, without modifier masks.
	Key physical_keycode = Key::NONE;
	uint32_t unicode = 0; ///unicode

	bool echo = false; /// true if this is an echo key

protected:
	static void _bind_methods();

public:
	void set_pressed(bool p_pressed);
	virtual bool is_pressed() const override;

	void set_keycode(Key p_keycode);
	Key get_keycode() const;

	void set_physical_keycode(Key p_keycode);
	Key get_physical_keycode() const;

	void set_unicode(char32_t p_unicode);
	char32_t get_unicode() const;

	void set_echo(bool p_enable);
	virtual bool is_echo() const override;

	Key get_keycode_with_modifiers() const;
	Key get_physical_keycode_with_modifiers() const;

	virtual bool action_match(const Ref<InputEvent> &p_event, bool *p_pressed, float *p_strength, float *p_raw_strength, float p_deadzone) const override;
	virtual bool is_match(const Ref<InputEvent> &p_event, bool p_exact_match = true) const override;

	virtual bool is_action_type() const override { return true; }

	virtual String as_text() const override;
	virtual String to_string() override;

	static Ref<InputEventKey> create_reference(Key p_keycode_with_modifier_masks);

	InputEventKey() {}
};

class InputEventMouse : public InputEventWithModifiers {
	GDCLASS(InputEventMouse, InputEventWithModifiers);

	MouseButton button_mask = MouseButton::NONE;

	Vector2 pos;
	Vector2 global_pos;

protected:
	static void _bind_methods();

public:
	void set_button_mask(MouseButton p_mask);
	MouseButton get_button_mask() const;

	void set_position(const Vector2 &p_pos);
	Vector2 get_position() const;

	void set_global_position(const Vector2 &p_global_pos);
	Vector2 get_global_position() const;

	InputEventMouse() {}
};

class InputEventMouseButton : public InputEventMouse {
	GDCLASS(InputEventMouseButton, InputEventMouse);

	float factor = 1;
	MouseButton button_index = MouseButton::NONE;
	bool pressed = false; //otherwise released
	bool double_click = false; //last even less than double click time

protected:
	static void _bind_methods();

public:
	void set_factor(float p_factor);
	float get_factor() const;

	void set_button_index(MouseButton p_index);
	MouseButton get_button_index() const;

	void set_pressed(bool p_pressed);
	virtual bool is_pressed() const override;

	void set_double_click(bool p_double_click);
	bool is_double_click() const;

	virtual Ref<InputEvent> xformed_by(const Transform2D &p_xform, const Vector2 &p_local_ofs = Vector2()) const override;

	virtual bool action_match(const Ref<InputEvent> &p_event, bool *p_pressed, float *p_strength, float *p_raw_strength, float p_deadzone) const override;
	virtual bool is_match(const Ref<InputEvent> &p_event, bool p_exact_match = true) const override;

	virtual bool is_action_type() const override { return true; }
	virtual String as_text() const override;
	virtual String to_string() override;

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
	virtual String to_string() override;

	virtual bool accumulate(const Ref<InputEvent> &p_event) override;

	InputEventMouseMotion() {}
};

class InputEventJoypadMotion : public InputEvent {
	GDCLASS(InputEventJoypadMotion, InputEvent);
	JoyAxis axis = (JoyAxis)0; ///< Joypad axis
	float axis_value = 0; ///< -1 to 1

protected:
	static void _bind_methods();

public:
	void set_axis(JoyAxis p_axis);
	JoyAxis get_axis() const;

	void set_axis_value(float p_value);
	float get_axis_value() const;

	virtual bool is_pressed() const override;

	virtual bool action_match(const Ref<InputEvent> &p_event, bool *p_pressed, float *p_strength, float *p_raw_strength, float p_deadzone) const override;
	virtual bool is_match(const Ref<InputEvent> &p_event, bool p_exact_match = true) const override;

	virtual bool is_action_type() const override { return true; }
	virtual String as_text() const override;
	virtual String to_string() override;

	InputEventJoypadMotion() {}
};

class InputEventJoypadButton : public InputEvent {
	GDCLASS(InputEventJoypadButton, InputEvent);

	JoyButton button_index = (JoyButton)0;
	bool pressed = false;
	float pressure = 0; //0 to 1
protected:
	static void _bind_methods();

public:
	void set_button_index(JoyButton p_index);
	JoyButton get_button_index() const;

	void set_pressed(bool p_pressed);
	virtual bool is_pressed() const override;

	void set_pressure(float p_pressure);
	float get_pressure() const;

	virtual bool action_match(const Ref<InputEvent> &p_event, bool *p_pressed, float *p_strength, float *p_raw_strength, float p_deadzone) const override;
	virtual bool is_match(const Ref<InputEvent> &p_event, bool p_exact_match = true) const override;

	virtual bool is_action_type() const override { return true; }

	virtual String as_text() const override;
	virtual String to_string() override;

	static Ref<InputEventJoypadButton> create_reference(JoyButton p_btn_index);

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
	virtual String to_string() override;

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
	virtual String to_string() override;

	virtual bool accumulate(const Ref<InputEvent> &p_event) override;

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

	virtual bool action_match(const Ref<InputEvent> &p_event, bool *p_pressed, float *p_strength, float *p_raw_strength, float p_deadzone) const override;
	virtual bool is_match(const Ref<InputEvent> &p_event, bool p_exact_match = true) const override;

	virtual bool is_action_type() const override { return true; }

	virtual String as_text() const override;
	virtual String to_string() override;

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
	virtual String to_string() override;

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
	virtual String to_string() override;

	InputEventPanGesture() {}
};

class InputEventMIDI : public InputEvent {
	GDCLASS(InputEventMIDI, InputEvent);

	int channel = 0;
	MIDIMessage message = MIDIMessage::NONE;
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

	void set_message(const MIDIMessage p_message);
	MIDIMessage get_message() const;

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
	virtual String to_string() override;

	InputEventMIDI() {}
};

class InputEventShortcut : public InputEvent {
	GDCLASS(InputEventShortcut, InputEvent);

	Ref<Shortcut> shortcut;

protected:
	static void _bind_methods();

public:
	void set_shortcut(Ref<Shortcut> p_shortcut);
	Ref<Shortcut> get_shortcut();
	virtual bool is_pressed() const override;

	virtual String as_text() const override;
	virtual String to_string() override;
};

#endif // INPUT_EVENT_H
