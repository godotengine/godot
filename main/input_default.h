#ifndef INPUT_DEFAULT_H
#define INPUT_DEFAULT_H

#include "os/input.h"

class InputDefault : public Input {

	OBJ_TYPE( InputDefault, Input );
	_THREAD_SAFE_CLASS_

	int mouse_button_mask;
	Set<int> keys_pressed;
	Set<int> joy_buttons_pressed;
	Map<int,float> joy_axis;
	Map<StringName,int> custom_action_press;
	Map<int, String> joy_names;
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

	SpeedTrack mouse_speed_track;

	RES custom_cursor;

public:

	virtual bool is_key_pressed(int p_scancode);
	virtual bool is_mouse_button_pressed(int p_button);
	virtual bool is_joy_button_pressed(int p_device, int p_button);
	virtual bool is_action_pressed(const StringName& p_action);

	virtual float get_joy_axis(int p_device,int p_axis);
	String get_joy_name(int p_idx);
	void joy_connection_changed(int p_idx, bool p_connected, String p_name);

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

	InputDefault();

};

#endif // INPUT_DEFAULT_H
