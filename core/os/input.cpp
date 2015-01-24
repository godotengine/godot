/*************************************************************************/
/*  input.cpp                                                            */
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
#include "input.h"
#include "input_map.h"
#include "os/os.h"
#include "globals.h"
Input *Input::singleton=NULL;

Input *Input::get_singleton() {

	return singleton;
}

void Input::set_mouse_mode(MouseMode p_mode) {
	ERR_FAIL_INDEX(p_mode,3);
	OS::get_singleton()->set_mouse_mode((OS::MouseMode)p_mode);
}

Input::MouseMode Input::get_mouse_mode() const {

	return (MouseMode)OS::get_singleton()->get_mouse_mode();
}

void Input::_bind_methods() {

	ObjectTypeDB::bind_method(_MD("is_key_pressed","scancode"),&Input::is_key_pressed);
	ObjectTypeDB::bind_method(_MD("is_mouse_button_pressed","button"),&Input::is_mouse_button_pressed);
	ObjectTypeDB::bind_method(_MD("is_joy_button_pressed","device","button"),&Input::is_joy_button_pressed);
	ObjectTypeDB::bind_method(_MD("is_action_pressed","action"),&Input::is_action_pressed);
	ObjectTypeDB::bind_method(_MD("get_joy_axis","device","axis"),&Input::get_joy_axis);
	ObjectTypeDB::bind_method(_MD("get_joy_name","device"),&Input::get_joy_name);
	ObjectTypeDB::bind_method(_MD("get_accelerometer"),&Input::get_accelerometer);
	ObjectTypeDB::bind_method(_MD("get_mouse_pos"),&Input::get_mouse_pos);
	ObjectTypeDB::bind_method(_MD("get_mouse_speed"),&Input::get_mouse_speed);
	ObjectTypeDB::bind_method(_MD("get_mouse_button_mask"),&Input::get_mouse_button_mask);
	ObjectTypeDB::bind_method(_MD("set_mouse_mode","mode"),&Input::set_mouse_mode);
	ObjectTypeDB::bind_method(_MD("get_mouse_mode"),&Input::get_mouse_mode);
	ObjectTypeDB::bind_method(_MD("warp_mouse_pos","to"),&Input::warp_mouse_pos);
	ObjectTypeDB::bind_method(_MD("action_press"),&Input::action_press);
	ObjectTypeDB::bind_method(_MD("action_release"),&Input::action_release);

	BIND_CONSTANT( MOUSE_MODE_VISIBLE );
	BIND_CONSTANT( MOUSE_MODE_HIDDEN );
	BIND_CONSTANT( MOUSE_MODE_CAPTURED );

	ADD_SIGNAL( MethodInfo("joy_connection_changed", PropertyInfo(Variant::INT, "index"), PropertyInfo(Variant::BOOL, "connected")) );
}

void Input::get_argument_options(const StringName& p_function,int p_idx,List<String>*r_options) const {
#ifdef TOOLS_ENABLED

	String pf=p_function;
	if (p_idx==0 && (pf=="is_action_pressed" || pf=="action_press" || pf=="action_release")) {

		List<PropertyInfo> pinfo;
		Globals::get_singleton()->get_property_list(&pinfo);

		for(List<PropertyInfo>::Element *E=pinfo.front();E;E=E->next()) {
			const PropertyInfo &pi=E->get();

			if (!pi.name.begins_with("input/"))
				continue;

			String name = pi.name.substr(pi.name.find("/")+1,pi.name.length());
			r_options->push_back("\""+name+"\"");

		}
	}
#endif

}

Input::Input() {

	singleton=this;
}


//////////////////////////////////////////////////////////


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

InputDefault::InputDefault() {

	mouse_button_mask=0;
	main_loop=NULL;
}
