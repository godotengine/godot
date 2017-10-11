/*************************************************************************/
/*  input.cpp                                                            */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
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
#include "project_settings.h"
Input *Input::singleton = NULL;

Input *Input::get_singleton() {

	return singleton;
}

void Input::set_mouse_mode(MouseMode p_mode) {
	ERR_FAIL_INDEX(p_mode, 4);
	OS::get_singleton()->set_mouse_mode((OS::MouseMode)p_mode);
}

Input::MouseMode Input::get_mouse_mode() const {

	return (MouseMode)OS::get_singleton()->get_mouse_mode();
}

void Input::_bind_methods() {

	ClassDB::bind_method(D_METHOD("is_key_pressed", "scancode"), &Input::is_key_pressed);
	ClassDB::bind_method(D_METHOD("is_mouse_button_pressed", "button"), &Input::is_mouse_button_pressed);
	ClassDB::bind_method(D_METHOD("is_joy_button_pressed", "device", "button"), &Input::is_joy_button_pressed);
	ClassDB::bind_method(D_METHOD("is_action_pressed", "action"), &Input::is_action_pressed);
	ClassDB::bind_method(D_METHOD("is_action_just_pressed", "action"), &Input::is_action_just_pressed);
	ClassDB::bind_method(D_METHOD("is_action_just_released", "action"), &Input::is_action_just_released);
	ClassDB::bind_method(D_METHOD("add_joy_mapping", "mapping", "update_existing"), &Input::add_joy_mapping, DEFVAL(false));
	ClassDB::bind_method(D_METHOD("remove_joy_mapping", "guid"), &Input::remove_joy_mapping);
	ClassDB::bind_method(D_METHOD("joy_connection_changed", "device", "connected", "name", "guid"), &Input::joy_connection_changed);
	ClassDB::bind_method(D_METHOD("is_joy_known", "device"), &Input::is_joy_known);
	ClassDB::bind_method(D_METHOD("get_joy_axis", "device", "axis"), &Input::get_joy_axis);
	ClassDB::bind_method(D_METHOD("get_joy_name", "device"), &Input::get_joy_name);
	ClassDB::bind_method(D_METHOD("get_joy_guid", "device"), &Input::get_joy_guid);
	ClassDB::bind_method(D_METHOD("get_connected_joypads"), &Input::get_connected_joypads);
	ClassDB::bind_method(D_METHOD("get_joy_vibration_strength", "device"), &Input::get_joy_vibration_strength);
	ClassDB::bind_method(D_METHOD("get_joy_vibration_duration", "device"), &Input::get_joy_vibration_duration);
	ClassDB::bind_method(D_METHOD("get_joy_button_string", "button_index"), &Input::get_joy_button_string);
	ClassDB::bind_method(D_METHOD("get_joy_button_index_from_string", "button"), &Input::get_joy_button_index_from_string);
	ClassDB::bind_method(D_METHOD("get_joy_axis_string", "axis_index"), &Input::get_joy_axis_string);
	ClassDB::bind_method(D_METHOD("get_joy_axis_index_from_string", "axis"), &Input::get_joy_axis_index_from_string);
	ClassDB::bind_method(D_METHOD("start_joy_vibration", "device", "weak_magnitude", "strong_magnitude", "duration"), &Input::start_joy_vibration, DEFVAL(0));
	ClassDB::bind_method(D_METHOD("stop_joy_vibration", "device"), &Input::stop_joy_vibration);
	ClassDB::bind_method(D_METHOD("get_gravity"), &Input::get_gravity);
	ClassDB::bind_method(D_METHOD("get_accelerometer"), &Input::get_accelerometer);
	ClassDB::bind_method(D_METHOD("get_magnetometer"), &Input::get_magnetometer);
	ClassDB::bind_method(D_METHOD("get_gyroscope"), &Input::get_gyroscope);
	//ClassDB::bind_method(D_METHOD("get_mouse_position"),&Input::get_mouse_position); - this is not the function you want
	ClassDB::bind_method(D_METHOD("get_last_mouse_speed"), &Input::get_last_mouse_speed);
	ClassDB::bind_method(D_METHOD("get_mouse_button_mask"), &Input::get_mouse_button_mask);
	ClassDB::bind_method(D_METHOD("set_mouse_mode", "mode"), &Input::set_mouse_mode);
	ClassDB::bind_method(D_METHOD("get_mouse_mode"), &Input::get_mouse_mode);
	ClassDB::bind_method(D_METHOD("warp_mouse_position", "to"), &Input::warp_mouse_position);
	ClassDB::bind_method(D_METHOD("action_press", "action"), &Input::action_press);
	ClassDB::bind_method(D_METHOD("action_release", "action"), &Input::action_release);
	ClassDB::bind_method(D_METHOD("set_custom_mouse_cursor", "image", "hotspot"), &Input::set_custom_mouse_cursor, DEFVAL(Vector2()));
	ClassDB::bind_method(D_METHOD("parse_input_event", "event"), &Input::parse_input_event);

	BIND_ENUM_CONSTANT(MOUSE_MODE_VISIBLE);
	BIND_ENUM_CONSTANT(MOUSE_MODE_HIDDEN);
	BIND_ENUM_CONSTANT(MOUSE_MODE_CAPTURED);
	BIND_ENUM_CONSTANT(MOUSE_MODE_CONFINED);

	ADD_SIGNAL(MethodInfo("joy_connection_changed", PropertyInfo(Variant::INT, "index"), PropertyInfo(Variant::BOOL, "connected")));
}

void Input::get_argument_options(const StringName &p_function, int p_idx, List<String> *r_options) const {
#ifdef TOOLS_ENABLED

	String pf = p_function;
	if (p_idx == 0 && (pf == "is_action_pressed" || pf == "action_press" || pf == "action_release" || pf == "is_action_just_pressed" || pf == "is_action_just_released")) {

		List<PropertyInfo> pinfo;
		ProjectSettings::get_singleton()->get_property_list(&pinfo);

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
