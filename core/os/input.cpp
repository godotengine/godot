/*************************************************************************/
/*  input.cpp                                                            */
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

#include "input.h"

#include "core/input_map.h"
#include "core/os/os.h"
#include "core/project_settings.h"

#ifdef TOOLS_ENABLED
#include "editor/editor_settings.h"
#endif

Input *Input::singleton = nullptr;

Input *Input::get_singleton() {
	return singleton;
}

void Input::set_mouse_mode(MouseMode p_mode) {
	ERR_FAIL_INDEX((int)p_mode, 5);
	OS::get_singleton()->set_mouse_mode((OS::MouseMode)p_mode);
}

Input::MouseMode Input::get_mouse_mode() const {
	return (MouseMode)OS::get_singleton()->get_mouse_mode();
}

void Input::_bind_methods() {
	ClassDB::bind_method(D_METHOD("is_key_pressed", "scancode"), &Input::is_key_pressed);
	ClassDB::bind_method(D_METHOD("is_physical_key_pressed", "scancode"), &Input::is_physical_key_pressed);
	ClassDB::bind_method(D_METHOD("is_mouse_button_pressed", "button"), &Input::is_mouse_button_pressed);
	ClassDB::bind_method(D_METHOD("is_joy_button_pressed", "device", "button"), &Input::is_joy_button_pressed);
	ClassDB::bind_method(D_METHOD("is_action_pressed", "action", "exact"), &Input::is_action_pressed, DEFVAL(false));
	ClassDB::bind_method(D_METHOD("is_action_just_pressed", "action", "exact"), &Input::is_action_just_pressed, DEFVAL(false));
	ClassDB::bind_method(D_METHOD("is_action_just_released", "action", "exact"), &Input::is_action_just_released, DEFVAL(false));
	ClassDB::bind_method(D_METHOD("get_action_strength", "action", "exact"), &Input::get_action_strength, DEFVAL(false));
	ClassDB::bind_method(D_METHOD("get_action_raw_strength", "action", "exact"), &Input::get_action_raw_strength, DEFVAL(false));
	ClassDB::bind_method(D_METHOD("get_axis", "negative_action", "positive_action"), &Input::get_axis);
	ClassDB::bind_method(D_METHOD("get_vector", "negative_x", "positive_x", "negative_y", "positive_y", "deadzone"), &Input::get_vector, DEFVAL(-1.0f));
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
	ClassDB::bind_method(D_METHOD("vibrate_handheld", "duration_ms"), &Input::vibrate_handheld, DEFVAL(500));
	ClassDB::bind_method(D_METHOD("get_gravity"), &Input::get_gravity);
	ClassDB::bind_method(D_METHOD("get_accelerometer"), &Input::get_accelerometer);
	ClassDB::bind_method(D_METHOD("get_magnetometer"), &Input::get_magnetometer);
	ClassDB::bind_method(D_METHOD("get_gyroscope"), &Input::get_gyroscope);
	ClassDB::bind_method(D_METHOD("set_gravity", "value"), &Input::set_gravity);
	ClassDB::bind_method(D_METHOD("set_accelerometer", "value"), &Input::set_accelerometer);
	ClassDB::bind_method(D_METHOD("set_magnetometer", "value"), &Input::set_magnetometer);
	ClassDB::bind_method(D_METHOD("set_gyroscope", "value"), &Input::set_gyroscope);
	//ClassDB::bind_method(D_METHOD("get_mouse_position"),&Input::get_mouse_position); - this is not the function you want
	ClassDB::bind_method(D_METHOD("get_last_mouse_speed"), &Input::get_last_mouse_speed);
	ClassDB::bind_method(D_METHOD("get_mouse_button_mask"), &Input::get_mouse_button_mask);
	ClassDB::bind_method(D_METHOD("set_mouse_mode", "mode"), &Input::set_mouse_mode);
	ClassDB::bind_method(D_METHOD("get_mouse_mode"), &Input::get_mouse_mode);
	ClassDB::bind_method(D_METHOD("warp_mouse_position", "to"), &Input::warp_mouse_position);
	ClassDB::bind_method(D_METHOD("action_press", "action", "strength"), &Input::action_press, DEFVAL(1.f));
	ClassDB::bind_method(D_METHOD("action_release", "action"), &Input::action_release);
	ClassDB::bind_method(D_METHOD("set_default_cursor_shape", "shape"), &Input::set_default_cursor_shape, DEFVAL(CURSOR_ARROW));
	ClassDB::bind_method(D_METHOD("get_current_cursor_shape"), &Input::get_current_cursor_shape);
	ClassDB::bind_method(D_METHOD("set_custom_mouse_cursor", "image", "shape", "hotspot"), &Input::set_custom_mouse_cursor, DEFVAL(CURSOR_ARROW), DEFVAL(Vector2()));
	ClassDB::bind_method(D_METHOD("parse_input_event", "event"), &Input::parse_input_event);
	ClassDB::bind_method(D_METHOD("set_use_accumulated_input", "enable"), &Input::set_use_accumulated_input);
	ClassDB::bind_method(D_METHOD("is_using_accumulated_input"), &Input::is_using_accumulated_input);
	ClassDB::bind_method(D_METHOD("flush_buffered_events"), &Input::flush_buffered_events);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "mouse_mode"), "set_mouse_mode", "get_mouse_mode");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "use_accumulated_input"), "set_use_accumulated_input", "is_using_accumulated_input");

	BIND_ENUM_CONSTANT(MOUSE_MODE_VISIBLE);
	BIND_ENUM_CONSTANT(MOUSE_MODE_HIDDEN);
	BIND_ENUM_CONSTANT(MOUSE_MODE_CAPTURED);
	BIND_ENUM_CONSTANT(MOUSE_MODE_CONFINED);
	BIND_ENUM_CONSTANT(MOUSE_MODE_CONFINED_HIDDEN);

	BIND_ENUM_CONSTANT(CURSOR_ARROW);
	BIND_ENUM_CONSTANT(CURSOR_IBEAM);
	BIND_ENUM_CONSTANT(CURSOR_POINTING_HAND);
	BIND_ENUM_CONSTANT(CURSOR_CROSS);
	BIND_ENUM_CONSTANT(CURSOR_WAIT);
	BIND_ENUM_CONSTANT(CURSOR_BUSY);
	BIND_ENUM_CONSTANT(CURSOR_DRAG);
	BIND_ENUM_CONSTANT(CURSOR_CAN_DROP);
	BIND_ENUM_CONSTANT(CURSOR_FORBIDDEN);
	BIND_ENUM_CONSTANT(CURSOR_VSIZE);
	BIND_ENUM_CONSTANT(CURSOR_HSIZE);
	BIND_ENUM_CONSTANT(CURSOR_BDIAGSIZE);
	BIND_ENUM_CONSTANT(CURSOR_FDIAGSIZE);
	BIND_ENUM_CONSTANT(CURSOR_MOVE);
	BIND_ENUM_CONSTANT(CURSOR_VSPLIT);
	BIND_ENUM_CONSTANT(CURSOR_HSPLIT);
	BIND_ENUM_CONSTANT(CURSOR_HELP);

	ADD_SIGNAL(MethodInfo("joy_connection_changed", PropertyInfo(Variant::INT, "device"), PropertyInfo(Variant::BOOL, "connected")));
}

void Input::get_argument_options(const StringName &p_function, int p_idx, List<String> *r_options) const {
#ifdef TOOLS_ENABLED

	const String quote_style = EDITOR_GET("text_editor/completion/use_single_quotes") ? "'" : "\"";

	String pf = p_function;
	if (p_idx == 0 &&
			(pf == "is_action_pressed" || pf == "action_press" || pf == "action_release" ||
					pf == "is_action_just_pressed" || pf == "is_action_just_released" ||
					pf == "get_action_strength" || pf == "get_action_raw_strength" ||
					pf == "get_axis" || pf == "get_vector")) {
		List<PropertyInfo> pinfo;
		ProjectSettings::get_singleton()->get_property_list(&pinfo);

		for (List<PropertyInfo>::Element *E = pinfo.front(); E; E = E->next()) {
			const PropertyInfo &pi = E->get();

			if (!pi.name.begins_with("input/")) {
				continue;
			}

			String name = pi.name.substr(pi.name.find("/") + 1, pi.name.length());
			r_options->push_back(quote_style + name + quote_style);
		}
	}
#endif
}

Input::Input() {
	singleton = this;
}

//////////////////////////////////////////////////////////
