/**************************************************************************/
/*  input_event_spatial.mm                                                */
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

#include "input_event_spatial.h"

#include "core/object/class_db.h"

void InputEventSpatial::_bind_methods() {
	BIND_ENUM_CONSTANT(FLAG_HAS_CHIRALITY);
	BIND_ENUM_CONSTANT(FLAG_HAS_SELECTION_RAY);
	BIND_ENUM_CONSTANT(FLAG_MAX);

	BIND_ENUM_CONSTANT(CHIRALITY_LEFT);
	BIND_ENUM_CONSTANT(CHIRALITY_RIGHT);

	BIND_ENUM_CONSTANT(PHASE_ACTIVE);
	BIND_ENUM_CONSTANT(PHASE_CANCELLED);
	BIND_ENUM_CONSTANT(PHASE_ENDED);

	ClassDB::bind_method(D_METHOD("set_index", "index"), &InputEventSpatial::set_index);
	ClassDB::bind_method(D_METHOD("get_index"), &InputEventSpatial::get_index);
	ADD_PROPERTY(PropertyInfo(Variant::INT, "index"), "set_index", "get_index");

	ClassDB::bind_method(D_METHOD("set_phase", "phase"), &InputEventSpatial::set_phase);
	ClassDB::bind_method(D_METHOD("get_phase"), &InputEventSpatial::get_phase);
	ADD_PROPERTY(PropertyInfo(Variant::INT, "phase"), "set_phase", "get_phase");

	ClassDB::bind_method(D_METHOD("get_flag", "flag"), &InputEventSpatial::get_flag);
	ClassDB::bind_method(D_METHOD("set_flag", "flag", "value"), &InputEventSpatial::set_flag);
	ADD_PROPERTYI(PropertyInfo(Variant::BOOL, "has_chirality"), "set_flag", "get_flag", FLAG_HAS_CHIRALITY);
	ADD_PROPERTYI(PropertyInfo(Variant::BOOL, "has_selection_ray"), "set_flag", "get_flag", FLAG_HAS_SELECTION_RAY);

	ClassDB::bind_method(D_METHOD("get_chirality"), &InputEventSpatial::get_chirality);
	ClassDB::bind_method(D_METHOD("set_chirality", "chirality"), &InputEventSpatial::set_chirality);
	ADD_PROPERTY(PropertyInfo(Variant::INT, "chirality"), "set_chirality", "get_chirality");

	ClassDB::bind_method(D_METHOD("get_selection_ray_origin"), &InputEventSpatial::get_selection_ray_origin);
	ClassDB::bind_method(D_METHOD("set_selection_ray_origin", "selection_ray_origin"), &InputEventSpatial::set_selection_ray_origin);
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "selection_ray_origin"), "set_selection_ray_origin", "get_selection_ray_origin");

	ClassDB::bind_method(D_METHOD("get_selection_ray_direction"), &InputEventSpatial::get_selection_ray_direction);
	ClassDB::bind_method(D_METHOD("set_selection_ray_direction", "selection_ray_direction"), &InputEventSpatial::set_selection_ray_direction);
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "selection_ray_direction"), "set_selection_ray_direction", "get_selection_ray_direction");

	ClassDB::bind_method(D_METHOD("get_input_device_pose_position"), &InputEventSpatial::get_input_device_pose_position);
	ClassDB::bind_method(D_METHOD("set_input_device_pose_position", "input_device_pose_position"), &InputEventSpatial::set_input_device_pose_position);
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "input_device_pose_position"), "set_input_device_pose_position", "get_input_device_pose_position");

	ClassDB::bind_method(D_METHOD("get_input_device_pose_rotation"), &InputEventSpatial::get_input_device_pose_rotation);
	ClassDB::bind_method(D_METHOD("set_input_device_pose_rotation", "input_device_pose_rotation"), &InputEventSpatial::set_input_device_pose_rotation);
	ADD_PROPERTY(PropertyInfo(Variant::QUATERNION, "input_device_pose_rotation"), "set_input_device_pose_rotation", "get_input_device_pose_rotation");
}

String InputEventSpatial::as_text() const {
	String phase_text;
	switch (phase) {
		case PHASE_ACTIVE:
			phase_text = RTR("active");
			break;
		case PHASE_CANCELLED:
			phase_text = RTR("cancelled");
			break;
		case PHASE_ENDED:
			phase_text = RTR("ended");
			break;
	}

	String chirality_text = (chirality == CHIRALITY_LEFT) ? RTR("left") : RTR("right");

	return vformat(RTR("Spatial event %s, %s hand, index %d"), phase_text, chirality_text, index);
}

String InputEventSpatial::_to_string() {
	String phase_string;
	switch (phase) {
		case PHASE_ACTIVE:
			phase_string = "PHASE_ACTIVE";
			break;
		case PHASE_CANCELLED:
			phase_string = "PHASE_CANCELLED";
			break;
		case PHASE_ENDED:
			phase_string = "PHASE_ENDED";
			break;
	}

	String chirality_string = (chirality == CHIRALITY_LEFT) ? "CHIRALITY_LEFT" : "CHIRALITY_RIGHT";
	String has_chirality = get_flag(FLAG_HAS_CHIRALITY) ? "true" : "false";
	String has_selection_ray = get_flag(FLAG_HAS_SELECTION_RAY) ? "true" : "false";

	return vformat("InputEventSpatial: index=%d, phase=%s, has_chirality=%s, chirality=%s, has_selection_ray=%s, selection_ray_origin=(%s), selection_ray_direction=(%s), input_device_position=(%s), input_device_rotation=(%s)",
			index, phase_string, has_chirality, chirality_string, has_selection_ray,
			String(selection_ray_origin), String(selection_ray_direction),
			String(input_device_pose_position), String(input_device_pose_rotation));
}
