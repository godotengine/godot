/*************************************************************************/
/*  input.cpp                                                            */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2019 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2019 Godot Engine contributors (cf. AUTHORS.md)    */
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
#include "globals.h"
#include "input_map.h"
#include "os/os.h"
Input *Input::singleton = NULL;

Input *Input::get_singleton() {

	return singleton;
}

void Input::set_mouse_mode(MouseMode p_mode) {
	ERR_FAIL_INDEX(p_mode, 3);
	OS::get_singleton()->set_mouse_mode((OS::MouseMode)p_mode);
}

Input::MouseMode Input::get_mouse_mode() const {

	return (MouseMode)OS::get_singleton()->get_mouse_mode();
}

void Input::_bind_methods() {

	ObjectTypeDB::bind_method(_MD("is_key_pressed", "scancode"), &Input::is_key_pressed);
	ObjectTypeDB::bind_method(_MD("is_mouse_button_pressed", "button"), &Input::is_mouse_button_pressed);
	ObjectTypeDB::bind_method(_MD("is_joy_button_pressed", "device", "button"), &Input::is_joy_button_pressed);
	ObjectTypeDB::bind_method(_MD("is_action_pressed", "action"), &Input::is_action_pressed);
	ObjectTypeDB::bind_method(_MD("add_joy_mapping", "mapping", "update_existing"), &Input::add_joy_mapping, DEFVAL(false));
	ObjectTypeDB::bind_method(_MD("remove_joy_mapping", "guid"), &Input::remove_joy_mapping);
	ObjectTypeDB::bind_method(_MD("is_joy_known", "device"), &Input::is_joy_known);
	ObjectTypeDB::bind_method(_MD("get_joy_axis", "device", "axis"), &Input::get_joy_axis);
	ObjectTypeDB::bind_method(_MD("get_joy_name", "device"), &Input::get_joy_name);
	ObjectTypeDB::bind_method(_MD("get_joy_guid", "device"), &Input::get_joy_guid);
	ObjectTypeDB::bind_method(_MD("get_connected_joysticks"), &Input::get_connected_joysticks);
	ObjectTypeDB::bind_method(_MD("get_joy_vibration_strength", "device"), &Input::get_joy_vibration_strength);
	ObjectTypeDB::bind_method(_MD("get_joy_vibration_duration", "device"), &Input::get_joy_vibration_duration);
	ObjectTypeDB::bind_method(_MD("get_joy_button_string", "button_index"), &Input::get_joy_button_string);
	ObjectTypeDB::bind_method(_MD("get_joy_button_index_from_string", "button"), &Input::get_joy_button_index_from_string);
	ObjectTypeDB::bind_method(_MD("get_joy_axis_string", "axis_index"), &Input::get_joy_axis_string);
	ObjectTypeDB::bind_method(_MD("get_joy_axis_index_from_string", "axis"), &Input::get_joy_axis_index_from_string);
	ObjectTypeDB::bind_method(_MD("start_joy_vibration", "device", "weak_magnitude", "strong_magnitude", "duration"), &Input::start_joy_vibration, DEFVAL(0));
	ObjectTypeDB::bind_method(_MD("stop_joy_vibration", "device"), &Input::stop_joy_vibration);
	ObjectTypeDB::bind_method(_MD("get_gravity"), &Input::get_gravity);
	ObjectTypeDB::bind_method(_MD("get_accelerometer"), &Input::get_accelerometer);
	ObjectTypeDB::bind_method(_MD("get_magnetometer"), &Input::get_magnetometer);
	ObjectTypeDB::bind_method(_MD("get_gyroscope"), &Input::get_gyroscope);
	//ObjectTypeDB::bind_method(_MD("get_mouse_pos"),&Input::get_mouse_pos); - this is not the function you want
	ObjectTypeDB::bind_method(_MD("get_mouse_speed"), &Input::get_mouse_speed);
	ObjectTypeDB::bind_method(_MD("get_mouse_button_mask"), &Input::get_mouse_button_mask);
	ObjectTypeDB::bind_method(_MD("set_mouse_mode", "mode"), &Input::set_mouse_mode);
	ObjectTypeDB::bind_method(_MD("get_mouse_mode"), &Input::get_mouse_mode);
	ObjectTypeDB::bind_method(_MD("warp_mouse_pos", "to"), &Input::warp_mouse_pos);
	ObjectTypeDB::bind_method(_MD("action_press", "action"), &Input::action_press);
	ObjectTypeDB::bind_method(_MD("action_release", "action"), &Input::action_release);
	ObjectTypeDB::bind_method(_MD("set_custom_mouse_cursor", "image:Texture", "shape", "hotspot"), &Input::set_custom_mouse_cursor, DEFVAL(CURSOR_ARROW), DEFVAL(Vector2()));
	ObjectTypeDB::bind_method(_MD("parse_input_event", "event"), &Input::parse_input_event);

	BIND_CONSTANT(MOUSE_MODE_VISIBLE);
	BIND_CONSTANT(MOUSE_MODE_HIDDEN);
	BIND_CONSTANT(MOUSE_MODE_CAPTURED);

	// Those constants are used to change the mouse texture for given cursor shapes
	BIND_CONSTANT(CURSOR_ARROW);
	BIND_CONSTANT(CURSOR_IBEAM);
	BIND_CONSTANT(CURSOR_POINTING_HAND);
	BIND_CONSTANT(CURSOR_CROSS);
	BIND_CONSTANT(CURSOR_WAIT);
	BIND_CONSTANT(CURSOR_BUSY);
	BIND_CONSTANT(CURSOR_DRAG);
	BIND_CONSTANT(CURSOR_CAN_DROP);
	BIND_CONSTANT(CURSOR_FORBIDDEN);
	BIND_CONSTANT(CURSOR_VSIZE);
	BIND_CONSTANT(CURSOR_HSIZE);
	BIND_CONSTANT(CURSOR_BDIAGSIZE);
	BIND_CONSTANT(CURSOR_FDIAGSIZE);
	BIND_CONSTANT(CURSOR_MOVE);
	BIND_CONSTANT(CURSOR_VSPLIT);
	BIND_CONSTANT(CURSOR_HSPLIT);
	BIND_CONSTANT(CURSOR_HELP);

	ADD_SIGNAL(MethodInfo("joy_connection_changed", PropertyInfo(Variant::INT, "index"), PropertyInfo(Variant::BOOL, "connected")));
}

void Input::get_argument_options(const StringName &p_function, int p_idx, List<String> *r_options) const {
#ifdef TOOLS_ENABLED

	String pf = p_function;
	if (p_idx == 0 && (pf == "is_action_pressed" || pf == "action_press" || pf == "action_release")) {

		List<PropertyInfo> pinfo;
		Globals::get_singleton()->get_property_list(&pinfo);

		for (List<PropertyInfo>::Element *E = pinfo.front(); E; E = E->next()) {
			const PropertyInfo &pi = E->get();

			if (!pi.name.begins_with("input/"))
				continue;

			String name = pi.name.substr(pi.name.find("/") + 1, pi.name.length());
			r_options->push_back("\"" + name + "\"");
		}
	}
#endif
}

Input::Input() {

	singleton = this;
}

//////////////////////////////////////////////////////////
