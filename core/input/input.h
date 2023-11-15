/**************************************************************************/
/*  input.h                                                               */
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

#ifndef INPUT_H
#define INPUT_H

#include "core/input/input_event.h"
#include "core/object/object.h"
#include "core/os/keyboard.h"
#include "core/os/thread_safe.h"
#include "core/templates/rb_set.h"
#include "core/variant/typed_array.h"

class Input : public Object {
	GDCLASS(Input, Object);
	_THREAD_SAFE_CLASS_

	static Input *singleton;

	static constexpr uint64_t MAX_EVENT = 31;

public:
	enum MouseMode {
		MOUSE_MODE_VISIBLE,
		MOUSE_MODE_HIDDEN,
		MOUSE_MODE_CAPTURED,
		MOUSE_MODE_CONFINED,
		MOUSE_MODE_CONFINED_HIDDEN,
	};

#undef CursorShape
	enum CursorShape {
		CURSOR_ARROW,
		CURSOR_IBEAM,
		CURSOR_POINTING_HAND,
		CURSOR_CROSS,
		CURSOR_WAIT,
		CURSOR_BUSY,
		CURSOR_DRAG,
		CURSOR_CAN_DROP,
		CURSOR_FORBIDDEN,
		CURSOR_VSIZE,
		CURSOR_HSIZE,
		CURSOR_BDIAGSIZE,
		CURSOR_FDIAGSIZE,
		CURSOR_MOVE,
		CURSOR_VSPLIT,
		CURSOR_HSPLIT,
		CURSOR_HELP,
		CURSOR_MAX
	};

	enum {
		JOYPADS_MAX = 16,
	};

	typedef void (*EventDispatchFunc)(const Ref<InputEvent> &p_event);

private:
	BitField<MouseButtonMask> mouse_button_mask;

	RBSet<Key> key_label_pressed;
	RBSet<Key> physical_keys_pressed;
	RBSet<Key> keys_pressed;
	RBSet<JoyButton> joy_buttons_pressed;
	RBMap<JoyAxis, float> _joy_axis;
	//RBMap<StringName,int> custom_action_press;
	Vector3 gravity;
	Vector3 accelerometer;
	Vector3 magnetometer;
	Vector3 gyroscope;
	Vector2 mouse_pos;
	int64_t mouse_window = 0;
	bool legacy_just_pressed_behavior = false;

	struct Action {
		uint64_t pressed_physics_frame = UINT64_MAX;
		uint64_t pressed_process_frame = UINT64_MAX;
		uint64_t released_physics_frame = UINT64_MAX;
		uint64_t released_process_frame = UINT64_MAX;
		uint64_t pressed = 0;
		bool exact = true;
		float strength = 0.0f;
		float raw_strength = 0.0f;
		LocalVector<float> strengths;
		LocalVector<float> raw_strengths;

		Action() {
			strengths.resize(MAX_EVENT + 1);
			raw_strengths.resize(MAX_EVENT + 1);

			for (uint64_t i = 0; i <= MAX_EVENT; i++) {
				strengths[i] = 0.0;
				raw_strengths[i] = 0.0;
			}
		}
	};

	HashMap<StringName, Action> action_state;

	bool emulate_touch_from_mouse = false;
	bool emulate_mouse_from_touch = false;
	bool use_input_buffering = false;
	bool use_accumulated_input = true;

	int mouse_from_touch_index = -1;

	struct VibrationInfo {
		float weak_magnitude;
		float strong_magnitude;
		float duration; // Duration in seconds
		uint64_t timestamp;
	};

	HashMap<int, VibrationInfo> joy_vibration;

	struct VelocityTrack {
		uint64_t last_tick = 0;
		Vector2 velocity;
		Vector2 accum;
		float accum_t = 0.0f;
		float min_ref_frame;
		float max_ref_frame;

		void update(const Vector2 &p_delta_p);
		void reset();
		VelocityTrack();
	};

	struct Joypad {
		StringName name;
		StringName uid;
		bool connected = false;
		bool last_buttons[(size_t)JoyButton::MAX] = { false };
		float last_axis[(size_t)JoyAxis::MAX] = { 0.0f };
		HatMask last_hat = HatMask::CENTER;
		int mapping = -1;
		int hat_current = 0;
		Dictionary info;
	};

	VelocityTrack mouse_velocity_track;
	HashMap<int, VelocityTrack> touch_velocity_track;
	HashMap<int, Joypad> joy_names;

	HashSet<uint32_t> ignored_device_ids;

	int fallback_mapping = -1;

	CursorShape default_shape = CURSOR_ARROW;

	enum JoyType {
		TYPE_BUTTON,
		TYPE_AXIS,
		TYPE_HAT,
		TYPE_MAX,
	};

	enum JoyAxisRange {
		NEGATIVE_HALF_AXIS = -1,
		FULL_AXIS = 0,
		POSITIVE_HALF_AXIS = 1
	};

	struct JoyEvent {
		int type = TYPE_MAX;
		int index = -1; // Can be either JoyAxis or JoyButton.
		float value = 0.f;
	};

	struct JoyBinding {
		JoyType inputType;
		union {
			JoyButton button;

			struct {
				JoyAxis axis;
				JoyAxisRange range;
				bool invert;
			} axis;

			struct {
				HatDir hat;
				HatMask hat_mask;
			} hat;

		} input;

		JoyType outputType;
		union {
			JoyButton button;

			struct {
				JoyAxis axis;
				JoyAxisRange range;
			} axis;

		} output;
	};

	struct JoyDeviceMapping {
		String uid;
		String name;
		Vector<JoyBinding> bindings;
	};

	Vector<JoyDeviceMapping> map_db;

	JoyEvent _get_mapped_button_event(const JoyDeviceMapping &mapping, JoyButton p_button);
	JoyEvent _get_mapped_axis_event(const JoyDeviceMapping &mapping, JoyAxis p_axis, float p_value, JoyAxisRange &r_range);
	void _get_mapped_hat_events(const JoyDeviceMapping &mapping, HatDir p_hat, JoyEvent r_events[(size_t)HatDir::MAX]);
	JoyButton _get_output_button(String output);
	JoyAxis _get_output_axis(String output);
	void _button_event(int p_device, JoyButton p_index, bool p_pressed);
	void _axis_event(int p_device, JoyAxis p_axis, float p_value);
	void _update_action_strength(Action &p_action, int p_event_index, float p_strength);
	void _update_action_raw_strength(Action &p_action, int p_event_index, float p_strength);

	void _parse_input_event_impl(const Ref<InputEvent> &p_event, bool p_is_emulated);

	List<Ref<InputEvent>> buffered_events;
#ifdef DEBUG_ENABLED
	HashSet<Ref<InputEvent>> frame_parsed_events;
	uint64_t last_parsed_frame = UINT64_MAX;
#endif

	friend class DisplayServer;

	static void (*set_mouse_mode_func)(MouseMode);
	static MouseMode (*get_mouse_mode_func)();
	static void (*warp_mouse_func)(const Vector2 &p_position);

	static CursorShape (*get_current_cursor_shape_func)();
	static void (*set_custom_mouse_cursor_func)(const Ref<Resource> &, CursorShape, const Vector2 &);

	EventDispatchFunc event_dispatch_function = nullptr;

protected:
	static void _bind_methods();

public:
	void set_mouse_mode(MouseMode p_mode);
	MouseMode get_mouse_mode() const;
	void get_argument_options(const StringName &p_function, int p_idx, List<String> *r_options) const override;

	static Input *get_singleton();

	bool is_anything_pressed() const;
	bool is_key_pressed(Key p_keycode) const;
	bool is_physical_key_pressed(Key p_keycode) const;
	bool is_key_label_pressed(Key p_keycode) const;
	bool is_mouse_button_pressed(MouseButton p_button) const;
	bool is_joy_button_pressed(int p_device, JoyButton p_button) const;
	bool is_action_pressed(const StringName &p_action, bool p_exact = false) const;
	bool is_action_just_pressed(const StringName &p_action, bool p_exact = false) const;
	bool is_action_just_released(const StringName &p_action, bool p_exact = false) const;
	float get_action_strength(const StringName &p_action, bool p_exact = false) const;
	float get_action_raw_strength(const StringName &p_action, bool p_exact = false) const;

	float get_axis(const StringName &p_negative_action, const StringName &p_positive_action) const;
	Vector2 get_vector(const StringName &p_negative_x, const StringName &p_positive_x, const StringName &p_negative_y, const StringName &p_positive_y, float p_deadzone = -1.0f) const;

	float get_joy_axis(int p_device, JoyAxis p_axis) const;
	String get_joy_name(int p_idx);
	TypedArray<int> get_connected_joypads();
	Vector2 get_joy_vibration_strength(int p_device);
	float get_joy_vibration_duration(int p_device);
	uint64_t get_joy_vibration_timestamp(int p_device);
	void joy_connection_changed(int p_idx, bool p_connected, String p_name, String p_guid = "", Dictionary p_joypad_info = Dictionary());

	Vector3 get_gravity() const;
	Vector3 get_accelerometer() const;
	Vector3 get_magnetometer() const;
	Vector3 get_gyroscope() const;

	Point2 get_mouse_position() const;
	Vector2 get_last_mouse_velocity();
	BitField<MouseButtonMask> get_mouse_button_mask() const;

	void warp_mouse(const Vector2 &p_position);
	Point2i warp_mouse_motion(const Ref<InputEventMouseMotion> &p_motion, const Rect2 &p_rect);

	void parse_input_event(const Ref<InputEvent> &p_event);

	void set_gravity(const Vector3 &p_gravity);
	void set_accelerometer(const Vector3 &p_accel);
	void set_magnetometer(const Vector3 &p_magnetometer);
	void set_gyroscope(const Vector3 &p_gyroscope);
	void set_joy_axis(int p_device, JoyAxis p_axis, float p_value);

	void start_joy_vibration(int p_device, float p_weak_magnitude, float p_strong_magnitude, float p_duration = 0);
	void stop_joy_vibration(int p_device);
	void vibrate_handheld(int p_duration_ms = 500);

	void set_mouse_position(const Point2 &p_posf);

	void action_press(const StringName &p_action, float p_strength = 1.f);
	void action_release(const StringName &p_action);

	void set_emulate_touch_from_mouse(bool p_emulate);
	bool is_emulating_touch_from_mouse() const;
	void ensure_touch_mouse_raised();

	void set_emulate_mouse_from_touch(bool p_emulate);
	bool is_emulating_mouse_from_touch() const;

	CursorShape get_default_cursor_shape() const;
	void set_default_cursor_shape(CursorShape p_shape);
	CursorShape get_current_cursor_shape() const;
	void set_custom_mouse_cursor(const Ref<Resource> &p_cursor, CursorShape p_shape = Input::CURSOR_ARROW, const Vector2 &p_hotspot = Vector2());

	void parse_mapping(String p_mapping);
	void joy_button(int p_device, JoyButton p_button, bool p_pressed);
	void joy_axis(int p_device, JoyAxis p_axis, float p_value);
	void joy_hat(int p_device, BitField<HatMask> p_val);

	void add_joy_mapping(String p_mapping, bool p_update_existing = false);
	void remove_joy_mapping(String p_guid);

	int get_unused_joy_id();

	bool is_joy_known(int p_device);
	String get_joy_guid(int p_device) const;
	bool should_ignore_device(int p_vendor_id, int p_product_id) const;
	Dictionary get_joy_info(int p_device) const;
	void set_fallback_mapping(String p_guid);

	void flush_buffered_events();
	bool is_using_input_buffering();
	void set_use_input_buffering(bool p_enable);
	void set_use_accumulated_input(bool p_enable);
	bool is_using_accumulated_input();

	void release_pressed_events();

	void set_event_dispatch_function(EventDispatchFunc p_function);

	Input();
	~Input();
};

VARIANT_ENUM_CAST(Input::MouseMode);
VARIANT_ENUM_CAST(Input::CursorShape);

#endif // INPUT_H
