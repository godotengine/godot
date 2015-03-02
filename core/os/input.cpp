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

String InputDefault::get_joy_name(int p_idx) {

	_THREAD_SAFE_METHOD_;
	return joy_names[p_idx].name;
};

static String _hex_str(uint8_t p_byte) {

	static const char* dict = "0123456789abcdef";
	char ret[3];
	ret[2] = 0;

	ret[0] = dict[p_byte>>4];
	ret[1] = dict[p_byte & 0xf];

	return ret;
};

void InputDefault::joy_connection_changed(int p_idx, bool p_connected, String p_name, String p_guid) {

	_THREAD_SAFE_METHOD_;
	Joystick js;
	js.name = p_connected ? p_name : "";
	js.uid = p_connected ? p_guid : "";
	js.mapping = -1;
	js.hat_current = 0;
	if (p_connected) {

		String uidname = p_guid;
		if (p_guid == "") {
			int uidlen = MIN(p_name.length(), 16);
			for (int i=0; i<uidlen; i++) {
				uidname = uidname + _hex_str(p_name[i]);
			};
		};
		js.uid = uidname;
	};
	joy_names[p_idx] = js;

	emit_signal("joy_connection_changed", p_idx, p_connected);
};

InputDefault::InputDefault() {

	mouse_button_mask=0;
	main_loop=NULL;
}


//////////////////////////////////////////////////////////

void InputPC::joy_connection_changed(int p_idx, bool p_connected, String p_name, String p_guid) {

	InputDefault::joy_connection_changed(p_idx, p_connected, p_name, p_guid);

	if (p_connected) {
		printf("looking for mappings for guid %ls\n", String(joy_names[p_idx].uid).c_str());
		for (int i=0; i < map_db.size(); i++) {
			if (joy_names[p_idx].uid == map_db[i].uid) {
				joy_names[p_idx].mapping = i;
				break;
			};
		};
	};
};

uint32_t InputPC::joy_button(uint32_t p_last_id, int p_device, int p_button, bool p_pressed) {

	_THREAD_SAFE_METHOD_;
	const Joystick& joy = joy_names[p_device];
printf("got button %i, mapping is %i\n", p_button, joy.mapping);
	if (joy.mapping == -1) {
		return _button_event(p_last_id, p_device, p_button, p_pressed);
	};

	Map<int,JoyEvent>::Element* el = map_db[joy.mapping].buttons.find(p_button);
	if (!el) {
		return _button_event(p_last_id, p_device, p_button, p_pressed);
	};

	JoyEvent map = el->get();
	if (map.type == TYPE_BUTTON) {

		return _button_event(p_last_id, p_device, map.index, p_pressed);
	};

	if (map.type == TYPE_AXIS) {
		return _axis_event(p_last_id, p_device, map.index, p_pressed ? 1.0 : 0.0);
	};

	return p_last_id; // no event?
};

uint32_t InputPC::joy_axis(uint32_t p_last_id, int p_device, int p_axis, float p_value) {

	_THREAD_SAFE_METHOD_;
	const Joystick& joy = joy_names[p_device];

	if (joy.mapping == -1) {
		return _axis_event(p_last_id, p_device, p_axis, p_value);
	};

	Map<int,JoyEvent>::Element* el = map_db[joy.mapping].axis.find(p_axis);
	if (!el) {
		return _axis_event(p_last_id, p_device, p_axis, p_value);
	};

	JoyEvent map = el->get();
	if (map.type == TYPE_BUTTON) {
		return _button_event(p_last_id, p_device, map.index, Math::abs(p_value) > 0.5 ? true : false);
	};

	if (map.type == TYPE_AXIS) {

		return _axis_event(p_last_id, p_device, map.index, p_value);
	};

	return p_last_id;
};

uint32_t InputPC::joy_hat(uint32_t p_last_id, int p_device, int p_val) {

	_THREAD_SAFE_METHOD_;
	const Joystick& joy = joy_names[p_device];

	JoyEvent* map;

	if (joy.mapping == -1) {
		map = hat_map_default;
	} else {
		map = map_db[joy.mapping].hat;
	};

	int cur_val = joy_names[p_device].hat_current;

	if ( (p_val & HAT_MASK_UP) != (cur_val & HAT_MASK_UP) ) {
		p_last_id = _button_event(p_last_id, p_device, map[HAT_UP].index, p_val & HAT_MASK_UP);
	};

	if ( (p_val & HAT_MASK_RIGHT) != (cur_val & HAT_MASK_RIGHT) ) {
		p_last_id = _button_event(p_last_id, p_device, map[HAT_RIGHT].index, p_val & HAT_MASK_RIGHT);
	};
	if ( (p_val & HAT_MASK_DOWN) != (cur_val & HAT_MASK_DOWN) ) {
		p_last_id = _button_event(p_last_id, p_device, map[HAT_DOWN].index, p_val & HAT_MASK_DOWN);
	};
	if ( (p_val & HAT_MASK_LEFT) != (cur_val & HAT_MASK_LEFT) ) {
		p_last_id = _button_event(p_last_id, p_device, map[HAT_LEFT].index, p_val & HAT_MASK_LEFT);
	};

	joy_names[p_device].hat_current = p_val;

	return p_last_id;
};

uint32_t InputPC::_button_event(uint32_t p_last_id, int p_device, int p_index, bool p_pressed) {

	InputEvent ievent;
	ievent.type = InputEvent::JOYSTICK_BUTTON;
	ievent.device = p_device;
	ievent.ID = ++p_last_id;
	ievent.joy_button.button_index = p_index;
	ievent.joy_button.pressed = p_pressed;

	parse_input_event(ievent);

	return p_last_id;
};

uint32_t InputPC::_axis_event(uint32_t p_last_id, int p_device, int p_axis, float p_value) {

	InputEvent ievent;
	ievent.type = InputEvent::JOYSTICK_MOTION;
	ievent.device = p_device;
	ievent.ID = ++p_last_id;
	ievent.joy_motion.axis = p_axis;
	ievent.joy_motion.axis_value = p_value;

	parse_input_event( ievent );

	return p_last_id;
};

static const char *s_ControllerMappings [] =
{
#ifdef WINDOWS_ENABLED
	"341a3608000000000000504944564944,Afterglow PS3 Controller,a:b1,b:b2,back:b8,dpdown:h0.4,dpleft:h0.8,dpright:h0.2,dpup:h0.1,guide:b12,leftshoulder:b4,leftstick:b10,lefttrigger:b6,leftx:a0,lefty:a1,rightshoulder:b5,rightstick:b11,righttrigger:b7,rightx:a2,righty:a3,start:b9,x:b0,y:b3,",
	"ffff0000000000000000504944564944,GameStop Gamepad,a:b0,b:b1,back:b8,dpdown:h0.4,dpleft:h0.8,dpright:h0.2,dpup:h0.1,guide:,leftshoulder:b4,leftstick:b10,lefttrigger:b6,leftx:a0,lefty:a1,rightshoulder:b5,rightstick:b11,righttrigger:b7,rightx:a2,righty:a3,start:b9,x:b2,y:b3,",
	"6d0416c2000000000000504944564944,Generic DirectInput Controller,a:b1,b:b2,back:b8,dpdown:h0.4,dpleft:h0.8,dpright:h0.2,dpup:h0.1,leftshoulder:b4,leftstick:b10,lefttrigger:b6,leftx:a0,lefty:a1,rightshoulder:b5,rightstick:b11,righttrigger:b7,rightx:a2,righty:a3,start:b9,x:b0,y:b3,",
	"6d0419c2000000000000504944564944,Logitech F710 Gamepad,a:b1,b:b2,back:b8,dpdown:h0.4,dpleft:h0.8,dpright:h0.2,dpup:h0.1,leftshoulder:b4,leftstick:b10,lefttrigger:b6,leftx:a0,lefty:a1,rightshoulder:b5,rightstick:b11,righttrigger:b7,rightx:a2,righty:a3,start:b9,x:b0,y:b3,", /* Guide button doesn't seem to be sent in DInput mode. */
	"4d6963726f736f66742050432d6a6f79,OUYA Controller,a:b0,b:b3,dpdown:b9,dpleft:b10,dpright:b11,dpup:b8,guide:b14,leftshoulder:b4,leftstick:b6,lefttrigger:b12,leftx:a0,lefty:a1,rightshoulder:b5,rightstick:b7,righttrigger:b13,rightx:a5,righty:a4,x:b1,y:b2,",
	"88880803000000000000504944564944,PS3 Controller,a:b2,b:b1,back:b8,dpdown:h0.8,dpleft:h0.4,dpright:h0.2,dpup:h0.1,guide:b12,leftshoulder:b4,leftstick:b9,lefttrigger:b6,leftx:a0,lefty:a1,rightshoulder:b5,rightstick:b10,righttrigger:b7,rightx:a3,righty:a4,start:b11,x:b0,y:b3,",
	"4c056802000000000000504944564944,PS3 Controller,a:b14,b:b13,back:b0,dpdown:b6,dpleft:b7,dpright:b5,dpup:b4,guide:b16,leftshoulder:b10,leftstick:b1,lefttrigger:b8,leftx:a0,lefty:a1,rightshoulder:b11,rightstick:b2,righttrigger:b9,rightx:a2,righty:a3,start:b3,x:b15,y:b12,",
	"25090500000000000000504944564944,PS3 DualShock,a:b2,b:b1,back:b9,dpdown:h0.8,dpleft:h0.4,dpright:h0.2,dpup:h0.1,guide:,leftshoulder:b6,leftstick:b10,lefttrigger:b4,leftx:a0,lefty:a1,rightshoulder:b7,rightstick:b11,righttrigger:b5,rightx:a2,righty:a3,start:b8,x:b0,y:b3,",
	"4c05c405000000000000504944564944,PS4 Controller,a:b1,b:b2,back:b8,dpdown:h0.4,dpleft:h0.8,dpright:h0.2,dpup:h0.1,guide:b12,leftshoulder:b4,leftstick:b10,lefttrigger:a3,leftx:a0,lefty:a1,rightshoulder:b5,rightstick:b11,righttrigger:a4,rightx:a2,righty:a5,start:b9,x:b0,y:b3,",
#endif
#ifdef OSX_ENABLED
	"0500000047532047616d657061640000,GameStop Gamepad,a:b0,b:b1,back:b8,dpdown:h0.4,dpleft:h0.8,dpright:h0.2,dpup:h0.1,guide:,leftshoulder:b4,leftstick:b10,lefttrigger:b6,leftx:a0,lefty:a1,rightshoulder:b5,rightstick:b11,righttrigger:b7,rightx:a2,righty:a3,start:b9,x:b2,y:b3,",
	"6d0400000000000016c2000000000000,Logitech F310 Gamepad (DInput),a:b1,b:b2,back:b8,dpdown:h0.4,dpleft:h0.8,dpright:h0.2,dpup:h0.1,leftshoulder:b4,leftstick:b10,lefttrigger:b6,leftx:a0,lefty:a1,rightshoulder:b5,rightstick:b11,righttrigger:b7,rightx:a2,righty:a3,start:b9,x:b0,y:b3,", /* Guide button doesn't seem to be sent in DInput mode. */
	"6d0400000000000018c2000000000000,Logitech F510 Gamepad (DInput),a:b1,b:b2,back:b8,dpdown:h0.4,dpleft:h0.8,dpright:h0.2,dpup:h0.1,leftshoulder:b4,leftstick:b10,lefttrigger:b6,leftx:a0,lefty:a1,rightshoulder:b5,rightstick:b11,righttrigger:b7,rightx:a2,righty:a3,start:b9,x:b0,y:b3,",
	"6d040000000000001fc2000000000000,Logitech F710 Gamepad (XInput),a:b0,b:b1,back:b9,dpdown:b12,dpleft:b13,dpright:b14,dpup:b11,guide:b10,leftshoulder:b4,leftstick:b6,lefttrigger:a2,leftx:a0,lefty:a1,rightshoulder:b5,rightstick:b7,righttrigger:a5,rightx:a3,righty:a4,start:b8,x:b2,y:b3,",
	"6d0400000000000019c2000000000000,Logitech Wireless Gamepad (DInput),a:b1,b:b2,back:b8,dpdown:h0.4,dpleft:h0.8,dpright:h0.2,dpup:h0.1,leftshoulder:b4,leftstick:b10,lefttrigger:b6,leftx:a0,lefty:a1,rightshoulder:b5,rightstick:b11,righttrigger:b7,rightx:a2,righty:a3,start:b9,x:b0,y:b3,", /* This includes F710 in DInput mode and the "Logitech Cordless RumblePad 2", at the very least. */
	"4c050000000000006802000000000000,PS3 Controller,a:b14,b:b13,back:b0,dpdown:b6,dpleft:b7,dpright:b5,dpup:b4,guide:b16,leftshoulder:b10,leftstick:b1,lefttrigger:b8,leftx:a0,lefty:a1,rightshoulder:b11,rightstick:b2,righttrigger:b9,rightx:a2,righty:a3,start:b3,x:b15,y:b12,",
	"4c05000000000000c405000000000000,PS4 Controller,a:b1,b:b2,back:b8,dpdown:h0.4,dpleft:h0.8,dpright:h0.2,dpup:h0.1,guide:b12,leftshoulder:b4,leftstick:b10,lefttrigger:a3,leftx:a0,lefty:a1,rightshoulder:b5,rightstick:b11,righttrigger:a4,rightx:a2,righty:a5,start:b9,x:b0,y:b3,",
	"5e040000000000008e02000000000000,X360 Controller,a:b0,b:b1,back:b9,dpdown:b12,dpleft:b13,dpright:b14,dpup:b11,guide:b10,leftshoulder:b4,leftstick:b6,lefttrigger:a2,leftx:a0,lefty:a1,rightshoulder:b5,rightstick:b7,righttrigger:a5,rightx:a3,righty:a4,start:b8,x:b2,y:b3,",
#endif
#if X11_ENABLED
	"0500000047532047616d657061640000,GameStop Gamepad,a:b0,b:b1,back:b8,dpdown:h0.4,dpleft:h0.8,dpright:h0.2,dpup:h0.1,guide:,leftshoulder:b4,leftstick:b10,lefttrigger:b6,leftx:a0,lefty:a1,rightshoulder:b5,rightstick:b11,righttrigger:b7,rightx:a2,righty:a3,start:b9,x:b2,y:b3,",
	"03000000ba2200002010000001010000,Jess Technology USB Game Controller,a:b2,b:b1,back:b8,dpdown:h0.4,dpleft:h0.8,dpright:h0.2,dpup:h0.1,guide:,leftshoulder:b4,lefttrigger:b6,leftx:a0,lefty:a1,rightshoulder:b5,righttrigger:b7,rightx:a3,righty:a2,start:b9,x:b3,y:b0,",
	"030000006d04000019c2000010010000,Logitech Cordless RumblePad 2,a:b1,b:b2,back:b8,dpdown:h0.4,dpleft:h0.8,dpright:h0.2,dpup:h0.1,guide:,leftshoulder:b4,leftstick:b10,lefttrigger:b6,leftx:a0,lefty:a1,rightshoulder:b5,rightstick:b11,righttrigger:b7,rightx:a2,righty:a3,start:b9,x:b0,y:b3,",
	"030000006d0400001dc2000014400000,Logitech F310 Gamepad (XInput),a:b0,b:b1,back:b6,dpdown:h0.4,dpleft:h0.8,dpright:h0.2,dpup:h0.1,guide:b8,leftshoulder:b4,leftstick:b9,lefttrigger:a2,leftx:a0,lefty:a1,rightshoulder:b5,rightstick:b10,righttrigger:a5,rightx:a3,righty:a4,start:b7,x:b2,y:b3,",
	"030000006d0400001ec2000020200000,Logitech F510 Gamepad (XInput),a:b0,b:b1,back:b6,dpdown:h0.4,dpleft:h0.8,dpright:h0.2,dpup:h0.1,guide:b8,leftshoulder:b4,leftstick:b9,lefttrigger:a2,leftx:a0,lefty:a1,rightshoulder:b5,rightstick:b10,righttrigger:a5,rightx:a3,righty:a4,start:b7,x:b2,y:b3,",
	"030000006d04000019c2000011010000,Logitech F710 Gamepad (DInput),a:b1,b:b2,back:b8,dpdown:h0.4,dpleft:h0.8,dpright:h0.2,dpup:h0.1,leftshoulder:b4,leftstick:b10,lefttrigger:b6,leftx:a0,lefty:a1,rightshoulder:b5,rightstick:b11,righttrigger:b7,rightx:a2,righty:a3,start:b9,x:b0,y:b3,", /* Guide button doesn't seem to be sent in DInput mode. */
	"030000006d0400001fc2000005030000,Logitech F710 Gamepad (XInput),a:b0,b:b1,back:b6,dpdown:h0.4,dpleft:h0.8,dpright:h0.2,dpup:h0.1,guide:b8,leftshoulder:b4,leftstick:b9,lefttrigger:a2,leftx:a0,lefty:a1,rightshoulder:b5,rightstick:b10,righttrigger:a5,rightx:a3,righty:a4,start:b7,x:b2,y:b3,",
	"050000003620000100000002010000,OUYA Game Controller,a:b0,b:b3,dpdown:b9,dpleft:b10,dpright:b11,dpup:b8,guide:b14,leftshoulder:b4,leftstick:b6,lefttrigger:a2,leftx:a0,lefty:a1,platform:Linux,rightshoulder:b5,rightstick:b7,righttrigger:a5,rightx:a3,righty:a4,x:b1,y:b2,",
	"030000004c0500006802000011010000,PS3 Controller,a:b14,b:b13,back:b0,dpdown:b6,dpleft:b7,dpright:b5,dpup:b4,guide:b16,leftshoulder:b10,leftstick:b1,lefttrigger:b8,leftx:a0,lefty:a1,rightshoulder:b11,rightstick:b2,righttrigger:b9,rightx:a2,righty:a3,start:b3,x:b15,y:b12,",
	"030000004c050000c405000011010000,PS4 Controller,a:b1,b:b2,back:b8,dpdown:h0.4,dpleft:h0.8,dpright:h0.2,dpup:h0.1,guide:b12,leftshoulder:b4,leftstick:b10,lefttrigger:a3,leftx:a0,lefty:a1,rightshoulder:b5,rightstick:b11,righttrigger:a4,rightx:a2,righty:a5,start:b9,x:b0,y:b3,",
	"050000004c050000c405000000010000,PS4 Controller,a:b1,b:b2,back:b8,dpdown:h0.4,dpleft:h0.8,dpright:h0.2,dpup:h0.1,guide:b12,leftshoulder:b4,leftstick:b10,lefttrigger:a3,leftx:a0,lefty:a1,rightshoulder:b5,rightstick:b11,righttrigger:a4,rightx:a2,righty:a5,start:b9,x:b0,y:b3,",
	"03000000de280000fc11000001000000,Steam Controller,a:b0,b:b1,back:b6,dpdown:h0.4,dpleft:h0.8,dpright:h0.2,dpup:h0.1,guide:b8,leftshoulder:b4,leftstick:b9,lefttrigger:a2,leftx:a0,lefty:a1,rightshoulder:b5,rightstick:b10,righttrigger:a5,rightx:a3,righty:a4,start:b7,x:b2,y:b3,",
	"03000000de280000ff11000001000000,Valve Streaming Gamepad,a:b0,b:b1,back:b6,dpdown:h0.4,dpleft:h0.8,dpright:h0.2,dpup:h0.1,guide:b8,leftshoulder:b4,leftstick:b9,lefttrigger:a2,leftx:a0,lefty:a1,rightshoulder:b5,rightstick:b10,righttrigger:a5,rightx:a3,righty:a4,start:b7,x:b2,y:b3,",
	"030000005e0400008e02000014010000,X360 Controller,a:b0,b:b1,back:b6,dpdown:h0.4,dpleft:h0.8,dpright:h0.2,dpup:h0.1,guide:b8,leftshoulder:b4,leftstick:b9,lefttrigger:a2,leftx:a0,lefty:a1,rightshoulder:b5,rightstick:b10,righttrigger:a5,rightx:a3,righty:a4,start:b7,x:b2,y:b3,",
	"030000005e0400008e02000010010000,X360 Controller,a:b0,b:b1,back:b6,dpdown:h0.4,dpleft:h0.8,dpright:h0.2,dpup:h0.1,guide:b8,leftshoulder:b4,leftstick:b9,lefttrigger:a2,leftx:a0,lefty:a1,rightshoulder:b5,rightstick:b10,righttrigger:a5,rightx:a3,righty:a4,start:b7,x:b2,y:b3,",
	"030000005e0400001907000000010000,X360 Wireless Controller,a:b0,b:b1,back:b6,dpdown:b14,dpleft:b11,dpright:b12,dpup:b13,guide:b8,leftshoulder:b4,leftstick:b9,lefttrigger:a2,leftx:a0,lefty:a1,rightshoulder:b5,rightstick:b10,righttrigger:a5,rightx:a3,righty:a4,start:b7,x:b2,y:b3,",
	"030000005e0400009102000007010000,X360 Wireless Controller,a:b0,b:b1,back:b6,dpdown:b14,dpleft:b11,dpright:b12,dpup:b13,guide:b8,leftshoulder:b4,leftstick:b9,lefttrigger:a2,leftx:a0,lefty:a1,rightshoulder:b5,rightstick:b10,righttrigger:a5,rightx:a3,righty:a4,start:b7,x:b2,y:b3,",
	"030000005e040000d102000001010000,Xbox One Wireless Controller,a:b0,b:b1,y:b3,x:b2,start:b7,guide:b8,back:b6,leftstick:b9,rightstick:b10,leftshoulder:b4,rightshoulder:b5,dpup:h0.1,dpleft:h0.8,dpdown:h0.4,dpright:h0.2,leftx:a0,lefty:a1,rightx:a3,righty:a4,lefttrigger:a2,righttrigger:a5,",
#endif
#if defined(__ANDROID__)
	"4e564944494120436f72706f72617469,NVIDIA Controller,a:b0,b:b1,dpdown:h0.4,dpleft:h0.8,dpright:h0.2,dpup:h0.1,leftshoulder:b9,leftstick:b7,lefttrigger:a4,leftx:a0,lefty:a1,rightshoulder:b10,rightstick:b8,righttrigger:a5,rightx:a2,righty:a3,start:b6,x:b2,y:b3,",
#endif
	NULL
};

static int char2int(char p_input)
{
  if(p_input >= '0' && p_input <= '9')
	return p_input - '0';
  if(p_input >= 'A' && p_input <= 'F')
	return p_input - 'A' + 10;
  if(p_input >= 'a' && p_input <= 'f')
	return p_input - 'a' + 10;
  return -1;
}

InputPC::JoyEvent InputPC::_find_to_event(String p_to) {

	// string names of the SDL buttons in the same order as input_event.h godot buttons
	static const char* buttons[] = {"b", "a", "x", "y", "leftshoulder", "rightshoulder", "lefttrigger", "righttrigger", "leftstick", "rightstick", "back", "start", "dpup", "dpdown", "dpleft", "dpright", NULL };

	// same for axis
	static const char* axis[] = {"leftx", "lefty", "rightx", "righty", NULL };

	JoyEvent ret;
	ret.type = -1;

	int i=0;
	while (buttons[i]) {

		if (p_to == buttons[i]) {
			ret.type = TYPE_BUTTON;
			ret.index = i;
			ret.value = 0;
			return ret;
		};
		++i;
	};

	i = 0;
	while (axis[i]) {

		if (p_to == axis[i]) {
			ret.type = TYPE_AXIS;
			ret.index = i;
			ret.value = 0;
			return ret;
		};
		++i;
	};

	return ret;
};

void InputPC::parse_mapping(String p_mapping) {

	_THREAD_SAFE_METHOD_;
	JoyDeviceMapping mapping;

	Vector<String> entry = p_mapping.split(",");
	CharString uid;
	uid.resize(17);

	mapping.uid = entry[0];

	int idx = 1;
	while (++idx < entry.size()) {

		if (entry[idx] == "")
			continue;

		String from = entry[idx].get_slice(":", 1);
		String to = entry[idx].get_slice(":", 0);

		JoyEvent to_event = _find_to_event(to);
		if (to_event.type == -1)
			continue;

		String etype = from.substr(0, 1);
		if (etype == "a") {

			int aid = from.substr(1, from.length()-1).to_int();
			mapping.axis[aid] = to_event;

		} else if (etype == "b") {

			int bid = from.substr(1, from.length()-1).to_int();
			mapping.buttons[bid] = to_event;

		} else if (etype == "h") {

			int hat_value = from.get_slice(".", 1).to_int();
			switch (hat_value) {
			case 1:
				mapping.hat[HAT_UP] = to_event;
				break;
			case 2:
				mapping.hat[HAT_RIGHT] = to_event;
				break;
			case 4:
				mapping.hat[HAT_DOWN] = to_event;
				break;
			case 8:
				mapping.hat[HAT_LEFT] = to_event;
				break;
			};
		};
	};
	map_db.push_back(mapping);
	printf("added mapping with uuid %ls\n", mapping.uid.c_str());
};

InputPC::InputPC() {

	hat_map_default[HAT_UP].type = TYPE_BUTTON;
	hat_map_default[HAT_UP].index = JOY_DPAD_UP;
	hat_map_default[HAT_UP].value = 0;

	hat_map_default[HAT_RIGHT].type = TYPE_BUTTON;
	hat_map_default[HAT_RIGHT].index = JOY_DPAD_RIGHT;
	hat_map_default[HAT_RIGHT].value = 0;

	hat_map_default[HAT_DOWN].type = TYPE_BUTTON;
	hat_map_default[HAT_DOWN].index = JOY_DPAD_DOWN;
	hat_map_default[HAT_DOWN].value = 0;

	hat_map_default[HAT_LEFT].type = TYPE_BUTTON;
	hat_map_default[HAT_LEFT].index = JOY_DPAD_LEFT;
	hat_map_default[HAT_LEFT].value = 0;

	int i = 0;
	while (s_ControllerMappings[i]) {

		parse_mapping(s_ControllerMappings[i++]);
	};

	String env_mapping = OS::get_singleton()->get_environment("SDL_GAMECONTROLLERCONFIG");
	if (env_mapping != "") {

		Vector<String> entries = env_mapping.split("\n");
		for (int i=0; i < entries.size(); i++) {
			if (entries[i] == "")
				continue;
			parse_mapping(entries[i]);
		};
	};
};

InputPC::~InputPC() {

};
