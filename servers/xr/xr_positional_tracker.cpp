/*************************************************************************/
/*  xr_positional_tracker.cpp                                            */
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

#include "xr_positional_tracker.h"

#include "core/input/input.h"

void XRPositionalTracker::_bind_methods() {
	BIND_ENUM_CONSTANT(TRACKER_HAND_UNKNOWN);
	BIND_ENUM_CONSTANT(TRACKER_HAND_LEFT);
	BIND_ENUM_CONSTANT(TRACKER_HAND_RIGHT);

	ClassDB::bind_method(D_METHOD("get_tracker_type"), &XRPositionalTracker::get_tracker_type);
	ClassDB::bind_method(D_METHOD("set_tracker_type", "type"), &XRPositionalTracker::set_tracker_type);
	ADD_PROPERTY(PropertyInfo(Variant::INT, "type"), "set_tracker_type", "get_tracker_type");

	ClassDB::bind_method(D_METHOD("get_tracker_name"), &XRPositionalTracker::get_tracker_name);
	ClassDB::bind_method(D_METHOD("set_tracker_name", "name"), &XRPositionalTracker::set_tracker_name);
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "name"), "set_tracker_name", "get_tracker_name");

	ClassDB::bind_method(D_METHOD("get_tracker_desc"), &XRPositionalTracker::get_tracker_desc);
	ClassDB::bind_method(D_METHOD("set_tracker_desc", "description"), &XRPositionalTracker::set_tracker_desc);
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "description"), "set_tracker_desc", "get_tracker_desc");

	ClassDB::bind_method(D_METHOD("get_tracker_hand"), &XRPositionalTracker::get_tracker_hand);
	ClassDB::bind_method(D_METHOD("set_tracker_hand", "hand"), &XRPositionalTracker::set_tracker_hand);
	ADD_PROPERTY(PropertyInfo(Variant::INT, "hand", PROPERTY_HINT_ENUM, "Unknown,Left,Right"), "set_tracker_hand", "get_tracker_hand");

	ClassDB::bind_method(D_METHOD("has_pose", "name"), &XRPositionalTracker::has_pose);
	ClassDB::bind_method(D_METHOD("get_pose", "name"), &XRPositionalTracker::get_pose);
	ClassDB::bind_method(D_METHOD("invalidate_pose", "name"), &XRPositionalTracker::invalidate_pose);
	ClassDB::bind_method(D_METHOD("set_pose", "name", "transform", "linear_velocity", "angular_velocity"), &XRPositionalTracker::set_pose);
	ADD_SIGNAL(MethodInfo("pose_changed", PropertyInfo(Variant::OBJECT, "pose", PROPERTY_HINT_RESOURCE_TYPE, "XRPose")));

	ClassDB::bind_method(D_METHOD("get_input", "name"), &XRPositionalTracker::get_input);
	ClassDB::bind_method(D_METHOD("set_input", "name", "value"), &XRPositionalTracker::set_input);
	ADD_SIGNAL(MethodInfo("button_pressed", PropertyInfo(Variant::STRING, "name")));
	ADD_SIGNAL(MethodInfo("button_released", PropertyInfo(Variant::STRING, "name")));
	ADD_SIGNAL(MethodInfo("input_value_changed", PropertyInfo(Variant::STRING, "name"), PropertyInfo(Variant::FLOAT, "value")));
	ADD_SIGNAL(MethodInfo("input_axis_changed", PropertyInfo(Variant::STRING, "name"), PropertyInfo(Variant::VECTOR2, "vector")));

	ClassDB::bind_method(D_METHOD("get_rumble"), &XRPositionalTracker::get_rumble);
	ClassDB::bind_method(D_METHOD("set_rumble", "rumble"), &XRPositionalTracker::set_rumble);
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "rumble"), "set_rumble", "get_rumble");
};

void XRPositionalTracker::set_tracker_type(XRServer::TrackerType p_type) {
	if (type != p_type) {
		type = p_type;
		hand = XRPositionalTracker::TRACKER_HAND_UNKNOWN;
	};
};

XRServer::TrackerType XRPositionalTracker::get_tracker_type() const {
	return type;
};

void XRPositionalTracker::set_tracker_name(const StringName &p_name) {
	// Note: this should not be changed after the tracker is registered with the XRServer!
	name = p_name;
};

StringName XRPositionalTracker::get_tracker_name() const {
	return name;
};

void XRPositionalTracker::set_tracker_desc(const String &p_desc) {
	description = p_desc;
}

String XRPositionalTracker::get_tracker_desc() const {
	return description;
}

XRPositionalTracker::TrackerHand XRPositionalTracker::get_tracker_hand() const {
	return hand;
};

void XRPositionalTracker::set_tracker_hand(const XRPositionalTracker::TrackerHand p_hand) {
	XRServer *xr_server = XRServer::get_singleton();
	ERR_FAIL_NULL(xr_server);

	if (hand != p_hand) {
		// we can only set this if we've previously set this to be a controller!!
		ERR_FAIL_COND((type != XRServer::TRACKER_CONTROLLER) && (p_hand != XRPositionalTracker::TRACKER_HAND_UNKNOWN));

		hand = p_hand;
	};
};

bool XRPositionalTracker::has_pose(const StringName &p_action_name) const {
	return poses.has(p_action_name);
}

Ref<XRPose> XRPositionalTracker::get_pose(const StringName &p_action_name) const {
	Ref<XRPose> pose;

	if (poses.has(p_action_name)) {
		pose = poses[p_action_name];
	}

	return pose;
}

void XRPositionalTracker::invalidate_pose(const StringName &p_action_name) {
	// only update this if we were tracking this pose
	if (poses.has(p_action_name)) {
		// We just set tracking data as invalid, we leave our current transform and velocity data as is so controllers don't suddenly jump to origin.
		poses[p_action_name]->set_has_tracking_data(false);
	}
}

void XRPositionalTracker::set_pose(const StringName &p_action_name, const Transform3D &p_transform, const Vector3 &p_linear_velocity, const Vector3 &p_angular_velocity) {
	Ref<XRPose> new_pose;

	new_pose.instantiate();
	new_pose->set_name(p_action_name);
	new_pose->set_has_tracking_data(true);
	new_pose->set_transform(p_transform);
	new_pose->set_linear_velocity(p_linear_velocity);
	new_pose->set_angular_velocity(p_angular_velocity);

	poses[p_action_name] = new_pose;
	emit_signal("pose_changed", new_pose);

	// TODO discuss whether we also want to create and emit an InputEventXRPose event
}

Variant XRPositionalTracker::get_input(const StringName &p_action_name) const {
	if (inputs.has(p_action_name)) {
		return inputs[p_action_name];
	} else {
		return Variant();
	}
}

void XRPositionalTracker::set_input(const StringName &p_action_name, const Variant &p_value) {
	bool changed = false;

	// XR inputs

	if (inputs.has(p_action_name)) {
		changed = inputs[p_action_name] != p_value;
	} else {
		changed = true;
	}

	if (changed) {
		// store the new value
		inputs[p_action_name] = p_value;

		// emit signals to let the rest of the world know
		switch (p_value.get_type()) {
			case Variant::BOOL: {
				bool pressed = p_value;
				if (pressed) {
					emit_signal("button_pressed", p_action_name);
				} else {
					emit_signal("button_released", p_action_name);
				}

				// TODO discuss whether we also want to create and emit an InputEventXRButton event
			} break;
			case Variant::FLOAT: {
				emit_signal("input_value_changed", p_action_name, p_value);

				// TODO discuss whether we also want to create and emit an InputEventXRValue event
			} break;
			case Variant::VECTOR2: {
				emit_signal("input_axis_changed", p_action_name, p_value);

				// TODO discuss whether we also want to create and emit an InputEventXRAxis event
			} break;
			default: {
				// ???
			} break;
		}
	}
}

real_t XRPositionalTracker::get_rumble() const {
	return rumble;
};

void XRPositionalTracker::set_rumble(real_t p_rumble) {
	if (p_rumble > 0.0) {
		rumble = p_rumble;
	} else {
		rumble = 0.0;
	};
};

XRPositionalTracker::XRPositionalTracker() {
	type = XRServer::TRACKER_UNKNOWN;
	name = "Unknown";
	hand = TRACKER_HAND_UNKNOWN;
	rumble = 0.0;
};
