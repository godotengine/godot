#ifndef INPUT_DEFAULT_H
#define INPUT_DEFAULT_H

#include "os/input.h"

class InputDefault : public Input {

	OBJ_TYPE( InputDefault, Input );
	_THREAD_SAFE_CLASS_

	int mouse_button_mask;
	Set<int> keys_pressed;
	Set<int> joy_buttons_pressed;
	Map<int,float> _joy_axis;
	Map<StringName,int> custom_action_press;
	Vector3 accelerometer;
	Vector2 mouse_pos;
	MainLoop *main_loop;

	bool emulate_touch;

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

	struct Joystick {
		StringName name;
		StringName uid;
		bool last_buttons[JOY_BUTTON_MAX];
		float last_axis[JOY_AXIS_MAX];
		float filter;
		int last_hat;
		int mapping;
		int hat_current;

		Joystick() {

			for (int i = 0; i < JOY_AXIS_MAX; i++) {

				last_axis[i] = 0.0f;

			}
			for (int i = 0; i < JOY_BUTTON_MAX; i++) {

				last_buttons[i] = false;
			}
			last_hat = HAT_MASK_CENTER;
			filter = 0.01f;
		}
	};

	SpeedTrack mouse_speed_track;
	Map<int, Joystick> joy_names;
	RES custom_cursor;
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
	struct JoyAxis {
		int min;
		float value;
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
	float _handle_deadzone(int p_device, int p_axis, float p_value);

public:



	virtual bool is_key_pressed(int p_scancode);
	virtual bool is_mouse_button_pressed(int p_button);
	virtual bool is_joy_button_pressed(int p_device, int p_button);
	virtual bool is_action_pressed(const StringName& p_action);

	virtual float get_joy_axis(int p_device,int p_axis);
	String get_joy_name(int p_idx);
	void joy_connection_changed(int p_idx, bool p_connected, String p_name, String p_guid = "");
	void parse_joystick_mapping(String p_mapping, bool p_update_existing);

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

	void set_emulate_touch(bool p_emulate);
	virtual bool is_emulating_touchscreen() const;

	virtual void set_custom_mouse_cursor(const RES& p_cursor,const Vector2& p_hotspot=Vector2());
	virtual void set_mouse_in_window(bool p_in_window);

	void parse_mapping(String p_mapping);
	uint32_t joy_button(uint32_t p_last_id, int p_device, int p_button, bool p_pressed);
	uint32_t joy_axis(uint32_t p_last_id, int p_device, int p_axis, const JoyAxis& p_value);
	uint32_t joy_hat(uint32_t p_last_id, int p_device, int p_val);

	InputDefault();
};

#endif // INPUT_DEFAULT_H
