#include "input_default.h"
#include "servers/visual_server.h"
#include "os/os.h"
#include "input_map.h"

void InputDefault::SpeedTrack::update(const Vector2& p_delta_p) {

	uint64_t tick = OS::get_singleton()->get_ticks_usec();
	uint32_t tdiff = tick-last_tick;
	float delta_t = tdiff / 1000000.0;
	last_tick=tick;


	accum+=p_delta_p;
	accum_t+=delta_t;

	if (accum_t>max_ref_frame*10)
		accum_t=max_ref_frame*10;

	while( accum_t>=min_ref_frame ) {

		float slice_t = min_ref_frame / accum_t;
		Vector2 slice = accum*slice_t;
		accum=accum-slice;
		accum_t-=min_ref_frame;

		speed=(slice/min_ref_frame).linear_interpolate(speed,min_ref_frame/max_ref_frame);
	}

}

void InputDefault::SpeedTrack::reset() {
	last_tick = OS::get_singleton()->get_ticks_usec();
	speed=Vector2();
	accum_t=0;
}

InputDefault::SpeedTrack::SpeedTrack() {

	 min_ref_frame=0.1;
	 max_ref_frame=0.3;
	 reset();
}

bool InputDefault::is_key_pressed(int p_scancode) {

	_THREAD_SAFE_METHOD_
	return keys_pressed.has(p_scancode);
}

bool InputDefault::is_mouse_button_pressed(int p_button) {

	_THREAD_SAFE_METHOD_
	return (mouse_button_mask&(1<<p_button))!=0;
}


static int _combine_device(int p_value,int p_device) {

	return p_value|(p_device<<20);
}

bool InputDefault::is_joy_button_pressed(int p_device, int p_button) {

	_THREAD_SAFE_METHOD_
	return joy_buttons_pressed.has(_combine_device(p_button,p_device));
}

bool InputDefault::is_action_pressed(const StringName& p_action) {

	if (custom_action_press.has(p_action))
		return true; //simpler

	const List<InputEvent> *alist = InputMap::get_singleton()->get_action_list(p_action);
	if (!alist)
		return NULL;


	for (const List<InputEvent>::Element *E=alist->front();E;E=E->next()) {


		int device=E->get().device;

		switch(E->get().type) {

			case InputEvent::KEY: {

				const InputEventKey &iek=E->get().key;
				if ((keys_pressed.has(iek.scancode)))
					return true;
			} break;
			case InputEvent::MOUSE_BUTTON: {

				const InputEventMouseButton &iemb=E->get().mouse_button;
				 if(mouse_button_mask&(1<<iemb.button_index))
					 return true;
			} break;
			case InputEvent::JOYSTICK_BUTTON: {

				const InputEventJoystickButton &iejb=E->get().joy_button;
				int c = _combine_device(iejb.button_index,device);
				if (joy_buttons_pressed.has(c))
					return true;
			} break;
		}
	}

	return false;
}

float InputDefault::get_joy_axis(int p_device,int p_axis) {

	_THREAD_SAFE_METHOD_
	int c = _combine_device(p_axis,p_device);
	if (joy_axis.has(c)) {
		return joy_axis[c];
	} else {
		return 0;
	}
}

String InputDefault::get_joy_name(int p_idx) {

	_THREAD_SAFE_METHOD_
	return joy_names[p_idx];
};

void InputDefault::joy_connection_changed(int p_idx, bool p_connected, String p_name) {

	_THREAD_SAFE_METHOD_
	joy_names[p_idx] = p_connected ? p_name : "";

	emit_signal("joy_connection_changed", p_idx, p_connected);
};

Vector3 InputDefault::get_accelerometer() {

	_THREAD_SAFE_METHOD_
	return accelerometer;
}

void InputDefault::parse_input_event(const InputEvent& p_event) {

	_THREAD_SAFE_METHOD_
	switch(p_event.type) {

		case InputEvent::KEY: {

			if (p_event.key.echo)
				break;
			if (p_event.key.scancode==0)
				break;

		//	print_line(p_event);

			if (p_event.key.pressed)
				keys_pressed.insert(p_event.key.scancode);
			else
				keys_pressed.erase(p_event.key.scancode);
		} break;
		case InputEvent::MOUSE_BUTTON: {

			if (p_event.mouse_button.doubleclick)
				break;

			if (p_event.mouse_button.pressed)
				mouse_button_mask|=(1<<p_event.mouse_button.button_index);
			else
				mouse_button_mask&=~(1<<p_event.mouse_button.button_index);

			if (main_loop && emulate_touch && p_event.mouse_button.button_index==1) {
				InputEventScreenTouch touch_event;
				touch_event.index=0;
				touch_event.pressed=p_event.mouse_button.pressed;
				touch_event.x=p_event.mouse_button.x;
				touch_event.y=p_event.mouse_button.y;
				InputEvent ev;
				ev.type=InputEvent::SCREEN_TOUCH;
				ev.screen_touch=touch_event;
				main_loop->input_event(ev);
			}
		} break;
		case InputEvent::MOUSE_MOTION: {

			if (main_loop && emulate_touch && p_event.mouse_motion.button_mask&1) {
				InputEventScreenDrag drag_event;
				drag_event.index=0;
				drag_event.x=p_event.mouse_motion.x;
				drag_event.y=p_event.mouse_motion.y;
				drag_event.relative_x=p_event.mouse_motion.relative_x;
				drag_event.relative_y=p_event.mouse_motion.relative_y;
				drag_event.speed_x=p_event.mouse_motion.speed_x;
				drag_event.speed_y=p_event.mouse_motion.speed_y;

				InputEvent ev;
				ev.type=InputEvent::SCREEN_DRAG;
				ev.screen_drag=drag_event;

				main_loop->input_event(ev);
			}

		} break;
		case InputEvent::JOYSTICK_BUTTON: {

			int c = _combine_device(p_event.joy_button.button_index,p_event.device);

			if (p_event.joy_button.pressed)
				joy_buttons_pressed.insert(c);
			else
				joy_buttons_pressed.erase(c);
		} break;
		case InputEvent::JOYSTICK_MOTION: {
			set_joy_axis(p_event.device, p_event.joy_motion.axis, p_event.joy_motion.axis_value);
		} break;

	}

	if (main_loop)
		main_loop->input_event(p_event);

}

void InputDefault::set_joy_axis(int p_device,int p_axis,float p_value) {

	_THREAD_SAFE_METHOD_
	int c = _combine_device(p_axis,p_device);
	joy_axis[c]=p_value;
}

void InputDefault::set_accelerometer(const Vector3& p_accel) {

	_THREAD_SAFE_METHOD_

	accelerometer=p_accel;

}

void InputDefault::set_main_loop(MainLoop *p_main_loop) {
	main_loop=p_main_loop;

}

void InputDefault::set_mouse_pos(const Point2& p_posf) {

	mouse_speed_track.update(p_posf-mouse_pos);
	mouse_pos=p_posf;
	if (custom_cursor.is_valid()) {
		VisualServer::get_singleton()->cursor_set_pos(get_mouse_pos());
	}
}

Point2 InputDefault::get_mouse_pos() const {

	return mouse_pos;
}
Point2 InputDefault::get_mouse_speed() const {

	return mouse_speed_track.speed;
}

int InputDefault::get_mouse_button_mask() const {

	return OS::get_singleton()->get_mouse_button_state();
}

void InputDefault::warp_mouse_pos(const Vector2& p_to) {

	OS::get_singleton()->warp_mouse_pos(p_to);
}


void InputDefault::iteration(float p_step) {


}

void InputDefault::action_press(const StringName& p_action) {

	if (custom_action_press.has(p_action)) {

		custom_action_press[p_action]++;
	} else {
		custom_action_press[p_action]=1;
	}
}

void InputDefault::action_release(const StringName& p_action){

	ERR_FAIL_COND(!custom_action_press.has(p_action));
	custom_action_press[p_action]--;
	if (custom_action_press[p_action]==0) {
		custom_action_press.erase(p_action);
	}
}

void InputDefault::set_emulate_touch(bool p_emulate) {

	emulate_touch=p_emulate;
}

bool InputDefault::is_emulating_touchscreen() const {

	return emulate_touch;
}

void InputDefault::set_custom_mouse_cursor(const RES& p_cursor,const Vector2& p_hotspot) {
	if (custom_cursor==p_cursor)
		return;

	custom_cursor=p_cursor;

	if (p_cursor.is_null()) {
		set_mouse_mode(MOUSE_MODE_VISIBLE);
		VisualServer::get_singleton()->cursor_set_visible(false);
	} else {
		set_mouse_mode(MOUSE_MODE_HIDDEN);
		VisualServer::get_singleton()->cursor_set_visible(true);
		VisualServer::get_singleton()->cursor_set_texture(custom_cursor->get_rid(),p_hotspot);
		VisualServer::get_singleton()->cursor_set_pos(get_mouse_pos());
	}
}

void InputDefault::set_mouse_in_window(bool p_in_window) {

	if (custom_cursor.is_valid()) {

		if (p_in_window) {
			set_mouse_mode(MOUSE_MODE_HIDDEN);
			VisualServer::get_singleton()->cursor_set_visible(true);
		} else {
			set_mouse_mode(MOUSE_MODE_VISIBLE);
			VisualServer::get_singleton()->cursor_set_visible(false);
		}

	}
}

InputDefault::InputDefault() {

	mouse_button_mask=0;
	emulate_touch=false;
	main_loop=NULL;
}
