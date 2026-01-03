/**************************************************************************/
/*  spring_bone_simulator_3d.cpp                                          */
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

#include "spring_bone_simulator_3d.h"
#include "spring_bone_simulator_3d.compat.inc"

#include "scene/3d/spring_bone_collision_3d.h"

// Original VRM Spring Bone movement logic was distributed by (c) VRM Consortium. Licensed under the MIT license.

bool SpringBoneSimulator3D::_set(const StringName &p_path, const Variant &p_value) {
	String path = p_path;

	if (path.begins_with("settings/")) {
		int which = path.get_slicec('/', 1).to_int();
		String what = path.get_slicec('/', 2);
		ERR_FAIL_INDEX_V(which, (int)settings.size(), false);

		if (what == "root_bone_name") {
			set_root_bone_name(which, p_value);
		} else if (what == "root_bone") {
			set_root_bone(which, p_value);
		} else if (what == "end_bone_name") {
			set_end_bone_name(which, p_value);
		} else if (what == "end_bone") {
			String opt = path.get_slicec('/', 3);
			if (opt.is_empty()) {
				set_end_bone(which, p_value);
			} else if (opt == "direction") {
				set_end_bone_direction(which, static_cast<BoneDirection>((int)p_value));
			} else if (opt == "length") {
				set_end_bone_length(which, p_value);
			} else {
				return false;
			}
		} else if (what == "extend_end_bone") {
			set_extend_end_bone(which, p_value);
		} else if (what == "center_from") {
			set_center_from(which, static_cast<CenterFrom>((int)p_value));
		} else if (what == "center_node") {
			set_center_node(which, p_value);
		} else if (what == "center_bone") {
			set_center_bone(which, p_value);
		} else if (what == "center_bone_name") {
			set_center_bone_name(which, p_value);
		} else if (what == "individual_config") {
			set_individual_config(which, p_value);
		} else if (what == "rotation_axis") {
			set_rotation_axis(which, static_cast<RotationAxis>((int)p_value));
		} else if (what == "rotation_axis_vector") {
			set_rotation_axis_vector(which, p_value);
		} else if (what == "radius") {
			String opt = path.get_slicec('/', 3);
			if (opt == "value") {
				set_radius(which, p_value);
			} else if (opt == "damping_curve") {
				set_radius_damping_curve(which, p_value);
			} else {
				return false;
			}
		} else if (what == "stiffness") {
			String opt = path.get_slicec('/', 3);
			if (opt == "value") {
				set_stiffness(which, p_value);
			} else if (opt == "damping_curve") {
				set_stiffness_damping_curve(which, p_value);
			} else {
				return false;
			}
		} else if (what == "drag") {
			String opt = path.get_slicec('/', 3);
			if (opt == "value") {
				set_drag(which, p_value);
			} else if (opt == "damping_curve") {
				set_drag_damping_curve(which, p_value);
			} else {
				return false;
			}
		} else if (what == "gravity") {
			String opt = path.get_slicec('/', 3);
			if (opt == "value") {
				set_gravity(which, p_value);
			} else if (opt == "damping_curve") {
				set_gravity_damping_curve(which, p_value);
			} else if (opt == "direction") {
				set_gravity_direction(which, p_value);
			} else {
				return false;
			}
		} else if (what == "enable_all_child_collisions") {
			set_enable_all_child_collisions(which, p_value);
		} else if (what == "joint_count") {
			set_joint_count(which, p_value);
		} else if (what == "joints") {
			int idx = path.get_slicec('/', 3).to_int();
			String prop = path.get_slicec('/', 4);
			if (prop == "rotation_axis") {
				set_joint_rotation_axis(which, idx, static_cast<RotationAxis>((int)p_value));
			} else if (prop == "rotation_axis_vector") {
				set_joint_rotation_axis_vector(which, idx, p_value);
			} else if (prop == "radius") {
				set_joint_radius(which, idx, p_value);
			} else if (prop == "stiffness") {
				set_joint_stiffness(which, idx, p_value);
			} else if (prop == "drag") {
				set_joint_drag(which, idx, p_value);
			} else if (prop == "gravity") {
				set_joint_gravity(which, idx, p_value);
			} else if (prop == "gravity_direction") {
				set_joint_gravity_direction(which, idx, p_value);
			} else {
				return false;
			}
		} else if (what == "exclude_collision_count") {
			set_exclude_collision_count(which, p_value);
		} else if (what == "exclude_collisions") {
			int idx = path.get_slicec('/', 3).to_int();
			set_exclude_collision_path(which, idx, p_value);
		} else if (what == "collision_count") {
			set_collision_count(which, p_value);
		} else if (what == "collisions") {
			int idx = path.get_slicec('/', 3).to_int();
			set_collision_path(which, idx, p_value);
		} else {
			return false;
		}
	}
	return true;
}

bool SpringBoneSimulator3D::_get(const StringName &p_path, Variant &r_ret) const {
	String path = p_path;

	if (path.begins_with("settings/")) {
		int which = path.get_slicec('/', 1).to_int();
		String what = path.get_slicec('/', 2);
		ERR_FAIL_INDEX_V(which, (int)settings.size(), false);

		if (what == "root_bone_name") {
			r_ret = get_root_bone_name(which);
		} else if (what == "root_bone") {
			r_ret = get_root_bone(which);
		} else if (what == "end_bone_name") {
			r_ret = get_end_bone_name(which);
		} else if (what == "end_bone") {
			String opt = path.get_slicec('/', 3);
			if (opt.is_empty()) {
				r_ret = get_end_bone(which);
			} else if (opt == "direction") {
				r_ret = (int)get_end_bone_direction(which);
			} else if (opt == "length") {
				r_ret = get_end_bone_length(which);
			} else {
				return false;
			}
		} else if (what == "extend_end_bone") {
			r_ret = is_end_bone_extended(which);
		} else if (what == "center_from") {
			r_ret = (int)get_center_from(which);
		} else if (what == "center_node") {
			r_ret = get_center_node(which);
		} else if (what == "center_bone") {
			r_ret = get_center_bone(which);
		} else if (what == "center_bone_name") {
			r_ret = get_center_bone_name(which);
		} else if (what == "individual_config") {
			r_ret = is_config_individual(which);
		} else if (what == "rotation_axis") {
			r_ret = (int)get_rotation_axis(which);
		} else if (what == "rotation_axis_vector") {
			r_ret = get_rotation_axis_vector(which);
		} else if (what == "radius") {
			String opt = path.get_slicec('/', 3);
			if (opt == "value") {
				r_ret = get_radius(which);
			} else if (opt == "damping_curve") {
				r_ret = get_radius_damping_curve(which);
			} else {
				return false;
			}
		} else if (what == "stiffness") {
			String opt = path.get_slicec('/', 3);
			if (opt == "value") {
				r_ret = get_stiffness(which);
			} else if (opt == "damping_curve") {
				r_ret = get_stiffness_damping_curve(which);
			} else {
				return false;
			}
		} else if (what == "drag") {
			String opt = path.get_slicec('/', 3);
			if (opt == "value") {
				r_ret = get_drag(which);
			} else if (opt == "damping_curve") {
				r_ret = get_drag_damping_curve(which);
			} else {
				return false;
			}
		} else if (what == "gravity") {
			String opt = path.get_slicec('/', 3);
			if (opt == "value") {
				r_ret = get_gravity(which);
			} else if (opt == "damping_curve") {
				r_ret = get_gravity_damping_curve(which);
			} else if (opt == "direction") {
				r_ret = get_gravity_direction(which);
			} else {
				return false;
			}
		} else if (what == "enable_all_child_collisions") {
			r_ret = are_all_child_collisions_enabled(which);
		} else if (what == "joint_count") {
			r_ret = get_joint_count(which);
		} else if (what == "joints") {
			int idx = path.get_slicec('/', 3).to_int();
			String prop = path.get_slicec('/', 4);
			if (prop == "bone_name") {
				r_ret = get_joint_bone_name(which, idx);
			} else if (prop == "bone") {
				r_ret = get_joint_bone(which, idx);
			} else if (prop == "rotation_axis") {
				r_ret = (int)get_joint_rotation_axis(which, idx);
			} else if (prop == "rotation_axis_vector") {
				r_ret = get_joint_rotation_axis_vector(which, idx);
			} else if (prop == "radius") {
				r_ret = get_joint_radius(which, idx);
			} else if (prop == "stiffness") {
				r_ret = get_joint_stiffness(which, idx);
			} else if (prop == "drag") {
				r_ret = get_joint_drag(which, idx);
			} else if (prop == "gravity") {
				r_ret = get_joint_gravity(which, idx);
			} else if (prop == "gravity_direction") {
				r_ret = get_joint_gravity_direction(which, idx);
			} else {
				return false;
			}
		} else if (what == "exclude_collision_count") {
			r_ret = get_exclude_collision_count(which);
		} else if (what == "exclude_collisions") {
			int idx = path.get_slicec('/', 3).to_int();
			r_ret = get_exclude_collision_path(which, idx);
		} else if (what == "collision_count") {
			r_ret = get_collision_count(which);
		} else if (what == "collisions") {
			int idx = path.get_slicec('/', 3).to_int();
			r_ret = get_collision_path(which, idx);
		} else {
			return false;
		}
	}
	return true;
}

void SpringBoneSimulator3D::_get_property_list(List<PropertyInfo> *p_list) const {
	String enum_hint;
	Skeleton3D *skeleton = get_skeleton();
	if (skeleton) {
		enum_hint = skeleton->get_concatenated_bone_names();
	}

	LocalVector<PropertyInfo> props;

	for (uint32_t i = 0; i < settings.size(); i++) {
		String path = "settings/" + itos(i) + "/";
		props.push_back(PropertyInfo(Variant::STRING, path + "root_bone_name", PROPERTY_HINT_ENUM_SUGGESTION, enum_hint));
		props.push_back(PropertyInfo(Variant::INT, path + "root_bone", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NO_EDITOR));
		props.push_back(PropertyInfo(Variant::STRING, path + "end_bone_name", PROPERTY_HINT_ENUM_SUGGESTION, enum_hint));
		props.push_back(PropertyInfo(Variant::INT, path + "end_bone", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NO_EDITOR));
		props.push_back(PropertyInfo(Variant::BOOL, path + "extend_end_bone"));
		props.push_back(PropertyInfo(Variant::INT, path + "end_bone/direction", PROPERTY_HINT_ENUM, SkeletonModifier3D::get_hint_bone_direction()));
		props.push_back(PropertyInfo(Variant::FLOAT, path + "end_bone/length", PROPERTY_HINT_RANGE, "0,1,0.001,or_greater,suffix:m"));
		props.push_back(PropertyInfo(Variant::INT, path + "center_from", PROPERTY_HINT_ENUM, "WorldOrigin,Node,Bone"));
		props.push_back(PropertyInfo(Variant::NODE_PATH, path + "center_node"));
		props.push_back(PropertyInfo(Variant::STRING, path + "center_bone_name", PROPERTY_HINT_ENUM_SUGGESTION, enum_hint));
		props.push_back(PropertyInfo(Variant::INT, path + "center_bone", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NO_EDITOR));
		props.push_back(PropertyInfo(Variant::BOOL, path + "individual_config"));
		props.push_back(PropertyInfo(Variant::INT, path + "rotation_axis", PROPERTY_HINT_ENUM, SkeletonModifier3D::get_hint_rotation_axis()));
		props.push_back(PropertyInfo(Variant::VECTOR3, path + "rotation_axis_vector"));
		props.push_back(PropertyInfo(Variant::FLOAT, path + "radius/value", PROPERTY_HINT_RANGE, "0,1,0.001,or_greater,suffix:m"));
		props.push_back(PropertyInfo(Variant::OBJECT, path + "radius/damping_curve", PROPERTY_HINT_RESOURCE_TYPE, "Curve"));
		props.push_back(PropertyInfo(Variant::FLOAT, path + "stiffness/value", PROPERTY_HINT_RANGE, "0,4,0.01,or_greater"));
		props.push_back(PropertyInfo(Variant::OBJECT, path + "stiffness/damping_curve", PROPERTY_HINT_RESOURCE_TYPE, "Curve"));
		props.push_back(PropertyInfo(Variant::FLOAT, path + "drag/value", PROPERTY_HINT_RANGE, "0,1,0.01,or_greater"));
		props.push_back(PropertyInfo(Variant::OBJECT, path + "drag/damping_curve", PROPERTY_HINT_RESOURCE_TYPE, "Curve"));
		props.push_back(PropertyInfo(Variant::FLOAT, path + "gravity/value", PROPERTY_HINT_RANGE, "0,1,0.01,or_greater,or_less,suffix:m/s"));
		props.push_back(PropertyInfo(Variant::OBJECT, path + "gravity/damping_curve", PROPERTY_HINT_RESOURCE_TYPE, "Curve"));
		props.push_back(PropertyInfo(Variant::VECTOR3, path + "gravity/direction"));
		props.push_back(PropertyInfo(Variant::INT, path + "joint_count", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_ARRAY, "Joints," + path + "joints/,static,const"));
		for (uint32_t j = 0; j < settings[i]->joints.size(); j++) {
			String joint_path = path + "joints/" + itos(j) + "/";
			props.push_back(PropertyInfo(Variant::STRING, joint_path + "bone_name", PROPERTY_HINT_ENUM_SUGGESTION, enum_hint, PROPERTY_USAGE_EDITOR | PROPERTY_USAGE_READ_ONLY));
			props.push_back(PropertyInfo(Variant::INT, joint_path + "bone", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_READ_ONLY));
			props.push_back(PropertyInfo(Variant::INT, joint_path + "rotation_axis", PROPERTY_HINT_ENUM, SkeletonModifier3D::get_hint_rotation_axis()));
			props.push_back(PropertyInfo(Variant::VECTOR3, joint_path + "rotation_axis_vector"));
			props.push_back(PropertyInfo(Variant::FLOAT, joint_path + "radius", PROPERTY_HINT_RANGE, "0,1,0.001,or_greater,suffix:m"));
			props.push_back(PropertyInfo(Variant::FLOAT, joint_path + "stiffness", PROPERTY_HINT_RANGE, "0,4,0.01,or_greater"));
			props.push_back(PropertyInfo(Variant::FLOAT, joint_path + "drag", PROPERTY_HINT_RANGE, "0,1,0.01,or_greater"));
			props.push_back(PropertyInfo(Variant::FLOAT, joint_path + "gravity", PROPERTY_HINT_RANGE, "0,1,0.01,or_greater,or_less,suffix:m/s"));
			props.push_back(PropertyInfo(Variant::VECTOR3, joint_path + "gravity_direction"));
		}
		props.push_back(PropertyInfo(Variant::BOOL, path + "enable_all_child_collisions"));
		props.push_back(PropertyInfo(Variant::INT, path + "exclude_collision_count", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_ARRAY, "Exclude Collisions," + path + "exclude_collisions/"));
		for (uint32_t j = 0; j < settings[i]->exclude_collisions.size(); j++) {
			String collision_path = path + "exclude_collisions/" + itos(j);
			props.push_back(PropertyInfo(Variant::NODE_PATH, collision_path, PROPERTY_HINT_NODE_PATH_VALID_TYPES, "SpringBoneCollision3D"));
		}
		props.push_back(PropertyInfo(Variant::INT, path + "collision_count", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_ARRAY, "Collisions," + path + "collisions/"));
		for (uint32_t j = 0; j < settings[i]->collisions.size(); j++) {
			String collision_path = path + "collisions/" + itos(j);
			props.push_back(PropertyInfo(Variant::NODE_PATH, collision_path, PROPERTY_HINT_NODE_PATH_VALID_TYPES, "SpringBoneCollision3D"));
		}
	}

	for (PropertyInfo &p : props) {
		_validate_dynamic_prop(p);
		p_list->push_back(p);
	}
}

void SpringBoneSimulator3D::_validate_dynamic_prop(PropertyInfo &p_property) const {
	PackedStringArray split = p_property.name.split("/");
	if (split.size() > 2 && split[0] == "settings") {
		int which = split[1].to_int();

		// Extended end bone option.
		bool force_hide = false;
		if (split[2] == "extend_end_bone" && get_end_bone(which) == -1) {
			p_property.usage = PROPERTY_USAGE_NONE;
			force_hide = true;
		}
		if (force_hide || (split[2] == "end_bone" && !is_end_bone_extended(which) && split.size() > 3)) {
			p_property.usage = PROPERTY_USAGE_NONE;
		}

		// Center option.
		if (get_center_from(which) != CENTER_FROM_BONE && (split[2] == "center_bone" || split[2] == "center_bone_name")) {
			p_property.usage = PROPERTY_USAGE_NONE;
		}
		if (get_center_from(which) != CENTER_FROM_NODE && split[2] == "center_node") {
			p_property.usage = PROPERTY_USAGE_NONE;
		}

		// Joints option.
		if (is_config_individual(which)) {
			if (split[2] == "rotation_axis" || split[2] == "rotation_axis_vector" || split[2] == "radius" || split[2] == "radius_damping_curve" ||
					split[2] == "stiffness" || split[2] == "stiffness_damping_curve" ||
					split[2] == "drag" || split[2] == "drag_damping_curve" ||
					split[2] == "gravity" || split[2] == "gravity_damping_curve" || split[2] == "gravity_direction") {
				p_property.usage = PROPERTY_USAGE_NONE;
			}
		} else {
			if (split[2] == "joints" || split[2] == "joint_count") {
				// Don't storage them since they are overridden by _update_joints().
				p_property.usage ^= PROPERTY_USAGE_STORAGE;
				p_property.usage |= PROPERTY_USAGE_READ_ONLY;
			}
			if (split[2] == "rotation_axis_vector" && get_rotation_axis(which) != ROTATION_AXIS_CUSTOM) {
				p_property.usage = PROPERTY_USAGE_NONE;
			}
		}

		// Collisions option.
		if (are_all_child_collisions_enabled(which)) {
			if (split[2] == "collisions" || split[2] == "collision_count") {
				p_property.usage = PROPERTY_USAGE_NONE;
			}
		} else {
			if (split[2] == "exclude_collisions" || split[2] == "exclude_collision_count") {
				p_property.usage = PROPERTY_USAGE_NONE;
			}
		}
	}
	if (split.size() > 3 && split[0] == "settings") {
		int which = split[1].to_int();
		int joint = split[3].to_int();
		// Joints option.
		if (split[2] == "joints" && split.size() > 4) {
			if (split[4] == "rotation_axis_vector" && get_joint_rotation_axis(which, joint) != ROTATION_AXIS_CUSTOM) {
				p_property.usage = PROPERTY_USAGE_NONE;
			}
		}
	}
}

void SpringBoneSimulator3D::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE: {
#ifdef TOOLS_ENABLED
			if (Engine::get_singleton()->is_editor_hint()) {
				set_notify_local_transform(true); // Used for updating gizmo in editor.
			}
#endif // TOOLS_ENABLED
			_make_collisions_dirty();
			_make_all_joints_dirty();
		} break;
#ifdef TOOLS_ENABLED
		case NOTIFICATION_LOCAL_TRANSFORM_CHANGED: {
			_make_gizmo_dirty();
		} break;
		case NOTIFICATION_EDITOR_PRE_SAVE: {
			saving = true;
		} break;
		case NOTIFICATION_EDITOR_POST_SAVE: {
			saving = false;
		} break;
#endif // TOOLS_ENABLED
	}
}

// Setting.

void SpringBoneSimulator3D::set_root_bone_name(int p_index, const String &p_bone_name) {
	ERR_FAIL_INDEX(p_index, (int)settings.size());
	settings[p_index]->root_bone_name = p_bone_name;
	Skeleton3D *sk = get_skeleton();
	if (sk) {
		set_root_bone(p_index, sk->find_bone(settings[p_index]->root_bone_name));
	}
}

String SpringBoneSimulator3D::get_root_bone_name(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, (int)settings.size(), String());
	return settings[p_index]->root_bone_name;
}

void SpringBoneSimulator3D::set_root_bone(int p_index, int p_bone) {
	ERR_FAIL_INDEX(p_index, (int)settings.size());
	bool changed = settings[p_index]->root_bone != p_bone;
	settings[p_index]->root_bone = p_bone;
	Skeleton3D *sk = get_skeleton();
	if (sk) {
		if (settings[p_index]->root_bone <= -1 || settings[p_index]->root_bone >= sk->get_bone_count()) {
			WARN_PRINT("Root bone index out of range!");
			settings[p_index]->root_bone = -1;
		} else {
			settings[p_index]->root_bone_name = sk->get_bone_name(settings[p_index]->root_bone);
		}
	}
	if (changed) {
		_update_joint_array(p_index);
	}
}

int SpringBoneSimulator3D::get_root_bone(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, (int)settings.size(), -1);
	return settings[p_index]->root_bone;
}

void SpringBoneSimulator3D::set_end_bone_name(int p_index, const String &p_bone_name) {
	ERR_FAIL_INDEX(p_index, (int)settings.size());
	settings[p_index]->end_bone_name = p_bone_name;
	Skeleton3D *sk = get_skeleton();
	if (sk) {
		set_end_bone(p_index, sk->find_bone(settings[p_index]->end_bone_name));
	}
}

String SpringBoneSimulator3D::get_end_bone_name(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, (int)settings.size(), String());
	return settings[p_index]->end_bone_name;
}

void SpringBoneSimulator3D::set_end_bone(int p_index, int p_bone) {
	ERR_FAIL_INDEX(p_index, (int)settings.size());
	bool changed = settings[p_index]->end_bone != p_bone;
	settings[p_index]->end_bone = p_bone;
	Skeleton3D *sk = get_skeleton();
	if (sk) {
		if (settings[p_index]->end_bone <= -1 || settings[p_index]->end_bone >= sk->get_bone_count()) {
			WARN_PRINT("End bone index out of range!");
			settings[p_index]->end_bone = -1;
		} else {
			settings[p_index]->end_bone_name = sk->get_bone_name(settings[p_index]->end_bone);
		}
	}
	if (changed) {
		_update_joint_array(p_index);
	}
	notify_property_list_changed();
}

int SpringBoneSimulator3D::get_end_bone(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, (int)settings.size(), -1);
	return settings[p_index]->end_bone;
}

void SpringBoneSimulator3D::set_extend_end_bone(int p_index, bool p_enabled) {
	ERR_FAIL_INDEX(p_index, (int)settings.size());
	settings[p_index]->extend_end_bone = p_enabled;
	_make_joints_dirty(p_index, true);
	notify_property_list_changed();
}

bool SpringBoneSimulator3D::is_end_bone_extended(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, (int)settings.size(), false);
	return settings[p_index]->extend_end_bone;
}

void SpringBoneSimulator3D::set_end_bone_direction(int p_index, BoneDirection p_bone_direction) {
	ERR_FAIL_INDEX(p_index, (int)settings.size());
	settings[p_index]->end_bone_direction = p_bone_direction;
#ifdef TOOLS_ENABLED
	_make_gizmo_dirty();
#endif // TOOLS_ENABLED
	if (mutable_bone_axes) {
		return; // Chain dir will be recaluclated in _update_bone_axis().
	}
	_make_joints_dirty(p_index, true);
}

SkeletonModifier3D::BoneDirection SpringBoneSimulator3D::get_end_bone_direction(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, (int)settings.size(), BONE_DIRECTION_FROM_PARENT);
	return settings[p_index]->end_bone_direction;
}

void SpringBoneSimulator3D::set_end_bone_length(int p_index, float p_length) {
	ERR_FAIL_INDEX(p_index, (int)settings.size());
	float old = settings[p_index]->end_bone_length;
	settings[p_index]->end_bone_length = p_length;
#ifdef TOOLS_ENABLED
	_make_gizmo_dirty();
#endif // TOOLS_ENABLED
	if (mutable_bone_axes && Math::is_zero_approx(old) == Math::is_zero_approx(p_length)) {
		return; // If chain size is not changed, length will be recaluclated in _update_bone_axis().
	}
	_make_joints_dirty(p_index, true);
}

float SpringBoneSimulator3D::get_end_bone_length(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, (int)settings.size(), 0);
	return settings[p_index]->end_bone_length;
}

Vector3 SpringBoneSimulator3D::get_end_bone_axis(int p_end_bone, BoneDirection p_direction) const {
	Vector3 axis;
	if (p_direction == BONE_DIRECTION_FROM_PARENT) {
		Skeleton3D *sk = get_skeleton();
		if (sk) {
			axis = sk->get_bone_rest(p_end_bone).basis.xform_inv(mutable_bone_axes ? sk->get_bone_pose(p_end_bone).origin : sk->get_bone_rest(p_end_bone).origin);
			axis.normalize();
		}
	} else {
		axis = get_vector_from_bone_axis(static_cast<BoneAxis>((int)p_direction));
	}
	return axis;
}

void SpringBoneSimulator3D::set_center_from(int p_index, CenterFrom p_center_from) {
	ERR_FAIL_INDEX(p_index, (int)settings.size());
	bool center_changed = settings[p_index]->center_from != p_center_from;
	settings[p_index]->center_from = p_center_from;
	if (center_changed) {
		reset();
	}
	notify_property_list_changed();
}

SpringBoneSimulator3D::CenterFrom SpringBoneSimulator3D::get_center_from(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, (int)settings.size(), CENTER_FROM_WORLD_ORIGIN);
	return settings[p_index]->center_from;
}

void SpringBoneSimulator3D::set_center_node(int p_index, const NodePath &p_node_path) {
	ERR_FAIL_INDEX(p_index, (int)settings.size());
	bool center_changed = settings[p_index]->center_node != p_node_path;
	settings[p_index]->center_node = p_node_path;
	if (center_changed) {
		reset();
	}
}

NodePath SpringBoneSimulator3D::get_center_node(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, (int)settings.size(), NodePath());
	return settings[p_index]->center_node;
}

void SpringBoneSimulator3D::set_center_bone_name(int p_index, const String &p_bone_name) {
	ERR_FAIL_INDEX(p_index, (int)settings.size());
	settings[p_index]->center_bone_name = p_bone_name;
	Skeleton3D *sk = get_skeleton();
	if (sk) {
		set_center_bone(p_index, sk->find_bone(settings[p_index]->center_bone_name));
	}
}

String SpringBoneSimulator3D::get_center_bone_name(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, (int)settings.size(), String());
	return settings[p_index]->center_bone_name;
}

void SpringBoneSimulator3D::set_center_bone(int p_index, int p_bone) {
	ERR_FAIL_INDEX(p_index, (int)settings.size());
	bool center_changed = settings[p_index]->center_bone != p_bone;
	settings[p_index]->center_bone = p_bone;
	Skeleton3D *sk = get_skeleton();
	if (sk) {
		if (settings[p_index]->center_bone <= -1 || settings[p_index]->center_bone >= sk->get_bone_count()) {
			WARN_PRINT("Center bone index out of range!");
			settings[p_index]->center_bone = -1;
		} else {
			settings[p_index]->center_bone_name = sk->get_bone_name(settings[p_index]->center_bone);
		}
	}
	if (center_changed) {
		reset();
	}
}

int SpringBoneSimulator3D::get_center_bone(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, (int)settings.size(), -1);
	return settings[p_index]->center_bone;
}

void SpringBoneSimulator3D::set_radius(int p_index, float p_radius) {
	ERR_FAIL_INDEX(p_index, (int)settings.size());
	if (is_config_individual(p_index)) {
		return; // Joint config is individual mode.
	}
	settings[p_index]->radius = p_radius;
	_make_joints_dirty(p_index);
}

float SpringBoneSimulator3D::get_radius(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, (int)settings.size(), 0);
	return settings[p_index]->radius;
}

void SpringBoneSimulator3D::set_radius_damping_curve(int p_index, const Ref<Curve> &p_damping_curve) {
	ERR_FAIL_INDEX(p_index, (int)settings.size());
	if (is_config_individual(p_index)) {
		return; // Joint config is individual mode.
	}
	if (settings[p_index]->radius_damping_curve.is_valid()) {
		settings[p_index]->radius_damping_curve->disconnect_changed(callable_mp(this, &SpringBoneSimulator3D::_make_joints_dirty));
	}
	settings[p_index]->radius_damping_curve = p_damping_curve;
	if (settings[p_index]->radius_damping_curve.is_valid()) {
		settings[p_index]->radius_damping_curve->connect_changed(callable_mp(this, &SpringBoneSimulator3D::_make_joints_dirty).bind(p_index));
	}
	_make_joints_dirty(p_index);
}

Ref<Curve> SpringBoneSimulator3D::get_radius_damping_curve(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, (int)settings.size(), Ref<Curve>());
	return settings[p_index]->radius_damping_curve;
}

void SpringBoneSimulator3D::set_stiffness(int p_index, float p_stiffness) {
	ERR_FAIL_INDEX(p_index, (int)settings.size());
	if (is_config_individual(p_index)) {
		return; // Joint config is individual mode.
	}
	settings[p_index]->stiffness = p_stiffness;
	_make_joints_dirty(p_index);
}

float SpringBoneSimulator3D::get_stiffness(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, (int)settings.size(), 0);
	return settings[p_index]->stiffness;
}

void SpringBoneSimulator3D::set_stiffness_damping_curve(int p_index, const Ref<Curve> &p_damping_curve) {
	ERR_FAIL_INDEX(p_index, (int)settings.size());
	if (is_config_individual(p_index)) {
		return; // Joint config is individual mode.
	}
	if (settings[p_index]->stiffness_damping_curve.is_valid()) {
		settings[p_index]->stiffness_damping_curve->disconnect_changed(callable_mp(this, &SpringBoneSimulator3D::_make_joints_dirty));
	}
	settings[p_index]->stiffness_damping_curve = p_damping_curve;
	if (settings[p_index]->stiffness_damping_curve.is_valid()) {
		settings[p_index]->stiffness_damping_curve->connect_changed(callable_mp(this, &SpringBoneSimulator3D::_make_joints_dirty).bind(p_index));
	}
	_make_joints_dirty(p_index);
}

Ref<Curve> SpringBoneSimulator3D::get_stiffness_damping_curve(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, (int)settings.size(), Ref<Curve>());
	return settings[p_index]->stiffness_damping_curve;
}

void SpringBoneSimulator3D::set_drag(int p_index, float p_drag) {
	ERR_FAIL_INDEX(p_index, (int)settings.size());
	if (is_config_individual(p_index)) {
		return; // Joint config is individual mode.
	}
	settings[p_index]->drag = p_drag;
	_make_joints_dirty(p_index);
}

float SpringBoneSimulator3D::get_drag(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, (int)settings.size(), 0);
	return settings[p_index]->drag;
}

void SpringBoneSimulator3D::set_drag_damping_curve(int p_index, const Ref<Curve> &p_damping_curve) {
	ERR_FAIL_INDEX(p_index, (int)settings.size());
	if (is_config_individual(p_index)) {
		return; // Joint config is individual mode.
	}
	if (settings[p_index]->drag_damping_curve.is_valid()) {
		settings[p_index]->drag_damping_curve->disconnect_changed(callable_mp(this, &SpringBoneSimulator3D::_make_joints_dirty));
	}
	settings[p_index]->drag_damping_curve = p_damping_curve;
	if (settings[p_index]->drag_damping_curve.is_valid()) {
		settings[p_index]->drag_damping_curve->connect_changed(callable_mp(this, &SpringBoneSimulator3D::_make_joints_dirty).bind(p_index));
	}
	_make_joints_dirty(p_index);
}

Ref<Curve> SpringBoneSimulator3D::get_drag_damping_curve(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, (int)settings.size(), Ref<Curve>());
	return settings[p_index]->drag_damping_curve;
}

void SpringBoneSimulator3D::set_gravity(int p_index, float p_gravity) {
	ERR_FAIL_INDEX(p_index, (int)settings.size());
	if (is_config_individual(p_index)) {
		return; // Joint config is individual mode.
	}
	settings[p_index]->gravity = p_gravity;
	_make_joints_dirty(p_index);
}

float SpringBoneSimulator3D::get_gravity(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, (int)settings.size(), 0);
	return settings[p_index]->gravity;
}

void SpringBoneSimulator3D::set_gravity_damping_curve(int p_index, const Ref<Curve> &p_damping_curve) {
	ERR_FAIL_INDEX(p_index, (int)settings.size());
	if (is_config_individual(p_index)) {
		return; // Joint config is individual mode.
	}
	if (settings[p_index]->gravity_damping_curve.is_valid()) {
		settings[p_index]->gravity_damping_curve->disconnect_changed(callable_mp(this, &SpringBoneSimulator3D::_make_joints_dirty));
	}
	settings[p_index]->gravity_damping_curve = p_damping_curve;
	if (settings[p_index]->gravity_damping_curve.is_valid()) {
		settings[p_index]->gravity_damping_curve->connect_changed(callable_mp(this, &SpringBoneSimulator3D::_make_joints_dirty).bind(p_index));
	}
	_make_joints_dirty(p_index);
}

Ref<Curve> SpringBoneSimulator3D::get_gravity_damping_curve(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, (int)settings.size(), Ref<Curve>());
	return settings[p_index]->gravity_damping_curve;
}

void SpringBoneSimulator3D::set_gravity_direction(int p_index, const Vector3 &p_gravity_direction) {
	ERR_FAIL_INDEX(p_index, (int)settings.size());
	ERR_FAIL_COND(p_gravity_direction.is_zero_approx());
	if (is_config_individual(p_index)) {
		return; // Joint config is individual mode.
	}
	settings[p_index]->gravity_direction = p_gravity_direction;
	_make_joints_dirty(p_index);
}

Vector3 SpringBoneSimulator3D::get_gravity_direction(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, (int)settings.size(), Vector3(0, -1, 0));
	return settings[p_index]->gravity_direction;
}

void SpringBoneSimulator3D::set_rotation_axis(int p_index, RotationAxis p_axis) {
	ERR_FAIL_INDEX(p_index, (int)settings.size());
	if (is_config_individual(p_index)) {
		return; // Joint config is individual mode.
	}
	settings[p_index]->rotation_axis = p_axis;
	_make_joints_dirty(p_index);
	notify_property_list_changed();
}

SkeletonModifier3D::RotationAxis SpringBoneSimulator3D::get_rotation_axis(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, (int)settings.size(), ROTATION_AXIS_ALL);
	return settings[p_index]->rotation_axis;
}

void SpringBoneSimulator3D::set_rotation_axis_vector(int p_index, const Vector3 &p_vector) {
	ERR_FAIL_INDEX(p_index, (int)settings.size());
	if (is_config_individual(p_index) || settings[p_index]->rotation_axis != ROTATION_AXIS_CUSTOM) {
		return; // Joint config is individual mode.
	}
	settings[p_index]->rotation_axis_vector = p_vector;
	_make_joints_dirty(p_index);
}

Vector3 SpringBoneSimulator3D::get_rotation_axis_vector(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, (int)settings.size(), Vector3());
	Vector3 ret;
	switch (settings[p_index]->rotation_axis) {
		case ROTATION_AXIS_X:
			ret = Vector3(1, 0, 0);
			break;
		case ROTATION_AXIS_Y:
			ret = Vector3(0, 1, 0);
			break;
		case ROTATION_AXIS_Z:
			ret = Vector3(0, 0, 1);
			break;
		case ROTATION_AXIS_ALL:
			ret = Vector3(0, 0, 0);
			break;
		case ROTATION_AXIS_CUSTOM:
			ret = settings[p_index]->rotation_axis_vector;
			break;
	}
	return ret;
}

void SpringBoneSimulator3D::set_setting_count(int p_count) {
	ERR_FAIL_COND(p_count < 0);

	int delta = p_count - (int)settings.size();
	if (delta < 0) {
		for (int i = delta; i < 0; i++) {
			memdelete(settings[(int)settings.size() + i]);
			settings[(int)settings.size() + i] = nullptr;
		}
	}
	settings.resize(p_count);
	delta++;
	if (delta > 1) {
		for (int i = 1; i < delta; i++) {
			settings[p_count - i] = memnew(SpringBone3DSetting);
		}
	}
	notify_property_list_changed();
}

int SpringBoneSimulator3D::get_setting_count() const {
	return settings.size();
}

void SpringBoneSimulator3D::clear_settings() {
	set_setting_count(0);
}

// Individual joints.

void SpringBoneSimulator3D::set_individual_config(int p_index, bool p_enabled) {
	ERR_FAIL_INDEX(p_index, (int)settings.size());
	settings[p_index]->individual_config = p_enabled;
	_make_joints_dirty(p_index, true);
	notify_property_list_changed();
}

bool SpringBoneSimulator3D::is_config_individual(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, (int)settings.size(), false);
	return settings[p_index]->individual_config;
}

String SpringBoneSimulator3D::get_joint_bone_name(int p_index, int p_joint) const {
	ERR_FAIL_INDEX_V(p_index, (int)settings.size(), String());
	const LocalVector<SpringBone3DJointSetting *> &joints = settings[p_index]->joints;
	ERR_FAIL_INDEX_V(p_joint, (int)joints.size(), String());
	return joints[p_joint]->bone_name;
}

void SpringBoneSimulator3D::_set_joint_bone(int p_index, int p_joint, int p_bone) {
	ERR_FAIL_INDEX(p_index, (int)settings.size());
	const LocalVector<SpringBone3DJointSetting *> &joints = settings[p_index]->joints;
	ERR_FAIL_INDEX(p_joint, (int)joints.size());
	joints[p_joint]->bone = p_bone;
	Skeleton3D *sk = get_skeleton();
	if (sk) {
		if (joints[p_joint]->bone <= -1 || joints[p_joint]->bone >= sk->get_bone_count()) {
			WARN_PRINT("Joint bone index out of range!");
			joints[p_joint]->bone = -1;
		} else {
			joints[p_joint]->bone_name = sk->get_bone_name(joints[p_joint]->bone);
		}
	}
}

int SpringBoneSimulator3D::get_joint_bone(int p_index, int p_joint) const {
	ERR_FAIL_INDEX_V(p_index, (int)settings.size(), -1);
	const LocalVector<SpringBone3DJointSetting *> &joints = settings[p_index]->joints;
	ERR_FAIL_INDEX_V(p_joint, (int)joints.size(), -1);
	return joints[p_joint]->bone;
}

void SpringBoneSimulator3D::set_joint_radius(int p_index, int p_joint, float p_radius) {
	ERR_FAIL_INDEX(p_index, (int)settings.size());
	if (!is_config_individual(p_index)) {
		return; // Joints are read-only.
	}
	const LocalVector<SpringBone3DJointSetting *> &joints = settings[p_index]->joints;
	ERR_FAIL_INDEX(p_joint, (int)joints.size());
	joints[p_joint]->radius = p_radius;
#ifdef TOOLS_ENABLED
	_make_gizmo_dirty();
#endif // TOOLS_ENABLED
}

float SpringBoneSimulator3D::get_joint_radius(int p_index, int p_joint) const {
	ERR_FAIL_INDEX_V(p_index, (int)settings.size(), 0);
	const LocalVector<SpringBone3DJointSetting *> &joints = settings[p_index]->joints;
	ERR_FAIL_INDEX_V(p_joint, (int)joints.size(), 0);
	return joints[p_joint]->radius;
}

void SpringBoneSimulator3D::set_joint_stiffness(int p_index, int p_joint, float p_stiffness) {
	ERR_FAIL_INDEX(p_index, (int)settings.size());
	if (!is_config_individual(p_index)) {
		return; // Joints are read-only.
	}
	const LocalVector<SpringBone3DJointSetting *> &joints = settings[p_index]->joints;
	ERR_FAIL_INDEX(p_joint, (int)joints.size());
	joints[p_joint]->stiffness = p_stiffness;
}

float SpringBoneSimulator3D::get_joint_stiffness(int p_index, int p_joint) const {
	ERR_FAIL_INDEX_V(p_index, (int)settings.size(), 0);
	const LocalVector<SpringBone3DJointSetting *> &joints = settings[p_index]->joints;
	ERR_FAIL_INDEX_V(p_joint, (int)joints.size(), 0);
	return joints[p_joint]->stiffness;
}

void SpringBoneSimulator3D::set_joint_drag(int p_index, int p_joint, float p_drag) {
	ERR_FAIL_INDEX(p_index, (int)settings.size());
	if (!is_config_individual(p_index)) {
		return; // Joints are read-only.
	}
	const LocalVector<SpringBone3DJointSetting *> &joints = settings[p_index]->joints;
	ERR_FAIL_INDEX(p_joint, (int)joints.size());
	joints[p_joint]->drag = p_drag;
}

float SpringBoneSimulator3D::get_joint_drag(int p_index, int p_joint) const {
	ERR_FAIL_INDEX_V(p_index, (int)settings.size(), 0);
	const LocalVector<SpringBone3DJointSetting *> &joints = settings[p_index]->joints;
	ERR_FAIL_INDEX_V(p_joint, (int)joints.size(), 0);
	return joints[p_joint]->drag;
}

void SpringBoneSimulator3D::set_joint_gravity(int p_index, int p_joint, float p_gravity) {
	ERR_FAIL_INDEX(p_index, (int)settings.size());
	if (!is_config_individual(p_index)) {
		return; // Joints are read-only.
	}
	const LocalVector<SpringBone3DJointSetting *> &joints = settings[p_index]->joints;
	ERR_FAIL_INDEX(p_joint, (int)joints.size());
	joints[p_joint]->gravity = p_gravity;
}

float SpringBoneSimulator3D::get_joint_gravity(int p_index, int p_joint) const {
	ERR_FAIL_INDEX_V(p_index, (int)settings.size(), 0);
	const LocalVector<SpringBone3DJointSetting *> &joints = settings[p_index]->joints;
	ERR_FAIL_INDEX_V(p_joint, (int)joints.size(), 0);
	return joints[p_joint]->gravity;
}

void SpringBoneSimulator3D::set_joint_gravity_direction(int p_index, int p_joint, const Vector3 &p_gravity_direction) {
	ERR_FAIL_INDEX(p_index, (int)settings.size());
	ERR_FAIL_COND(p_gravity_direction.is_zero_approx());
	if (!is_config_individual(p_index)) {
		return; // Joints are read-only.
	}
	const LocalVector<SpringBone3DJointSetting *> &joints = settings[p_index]->joints;
	ERR_FAIL_INDEX(p_joint, (int)joints.size());
	joints[p_joint]->gravity_direction = p_gravity_direction;
}

Vector3 SpringBoneSimulator3D::get_joint_gravity_direction(int p_index, int p_joint) const {
	ERR_FAIL_INDEX_V(p_index, (int)settings.size(), Vector3(0, -1, 0));
	const LocalVector<SpringBone3DJointSetting *> &joints = settings[p_index]->joints;
	ERR_FAIL_INDEX_V(p_joint, (int)joints.size(), Vector3(0, -1, 0));
	return joints[p_joint]->gravity_direction;
}

void SpringBoneSimulator3D::set_joint_rotation_axis(int p_index, int p_joint, RotationAxis p_axis) {
	ERR_FAIL_INDEX(p_index, (int)settings.size());
	if (!is_config_individual(p_index)) {
		return; // Joints are read-only.
	}
	const LocalVector<SpringBone3DJointSetting *> &joints = settings[p_index]->joints;
	ERR_FAIL_INDEX(p_joint, (int)joints.size());
	joints[p_joint]->rotation_axis = p_axis;
	Skeleton3D *sk = get_skeleton();
	if (sk) {
		_validate_rotation_axis(sk, p_index, p_joint);
	}
	notify_property_list_changed();
	settings[p_index]->simulation_dirty = true;
#ifdef TOOLS_ENABLED
	_make_gizmo_dirty();
#endif // TOOLS_ENABLED
}

SkeletonModifier3D::RotationAxis SpringBoneSimulator3D::get_joint_rotation_axis(int p_index, int p_joint) const {
	ERR_FAIL_INDEX_V(p_index, (int)settings.size(), ROTATION_AXIS_ALL);
	const LocalVector<SpringBone3DJointSetting *> &joints = settings[p_index]->joints;
	ERR_FAIL_INDEX_V(p_joint, (int)joints.size(), ROTATION_AXIS_ALL);
	return joints[p_joint]->rotation_axis;
}

void SpringBoneSimulator3D::set_joint_rotation_axis_vector(int p_index, int p_joint, const Vector3 &p_vector) {
	ERR_FAIL_INDEX(p_index, (int)settings.size());
	if (!is_config_individual(p_index) || settings[p_index]->rotation_axis != ROTATION_AXIS_CUSTOM) {
		return; // Joints are read-only.
	}
	const LocalVector<SpringBone3DJointSetting *> &joints = settings[p_index]->joints;
	ERR_FAIL_INDEX(p_joint, (int)joints.size());
	joints[p_joint]->rotation_axis_vector = p_vector;
	Skeleton3D *sk = get_skeleton();
	if (sk) {
		_validate_rotation_axis(sk, p_index, p_joint);
	}
	settings[p_index]->simulation_dirty = true;
#ifdef TOOLS_ENABLED
	_make_gizmo_dirty();
#endif // TOOLS_ENABLED
}

Vector3 SpringBoneSimulator3D::get_joint_rotation_axis_vector(int p_index, int p_joint) const {
	ERR_FAIL_INDEX_V(p_index, (int)settings.size(), Vector3());
	const LocalVector<SpringBone3DJointSetting *> &joints = settings[p_index]->joints;
	ERR_FAIL_INDEX_V(p_joint, (int)joints.size(), Vector3());
	return joints[p_joint]->get_rotation_axis_vector();
}

void SpringBoneSimulator3D::set_joint_count(int p_index, int p_count) {
	ERR_FAIL_INDEX(p_index, (int)settings.size());
	ERR_FAIL_COND(p_count < 0);
	LocalVector<SpringBone3DJointSetting *> &joints = settings[p_index]->joints;
	int delta = p_count - joints.size();
	if (delta < 0) {
		for (int i = delta; i < 0; i++) {
			memdelete(joints[joints.size() + i]);
			joints[joints.size() + i] = nullptr;
		}
	}
	joints.resize(p_count);
	delta++;
	if (delta > 1) {
		for (int i = 1; i < delta; i++) {
			joints[p_count - i] = memnew(SpringBone3DJointSetting);
		}
	}
	notify_property_list_changed();
}

int SpringBoneSimulator3D::get_joint_count(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, (int)settings.size(), 0);
	const LocalVector<SpringBone3DJointSetting *> &joints = settings[p_index]->joints;
	return joints.size();
}

// Individual collisions.

void SpringBoneSimulator3D::set_enable_all_child_collisions(int p_index, bool p_enabled) {
	ERR_FAIL_INDEX(p_index, (int)settings.size());
	settings[p_index]->enable_all_child_collisions = p_enabled;
	notify_property_list_changed();
}

bool SpringBoneSimulator3D::are_all_child_collisions_enabled(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, (int)settings.size(), false);
	return settings[p_index]->enable_all_child_collisions;
}

void SpringBoneSimulator3D::set_exclude_collision_path(int p_index, int p_collision, const NodePath &p_node_path) {
	ERR_FAIL_INDEX(p_index, (int)settings.size());
	if (!are_all_child_collisions_enabled(p_index)) {
		return; // Exclude collision list is disabled.
	}
	LocalVector<NodePath> &setting_exclude_collisions = settings[p_index]->exclude_collisions;
	ERR_FAIL_INDEX(p_collision, (int)setting_exclude_collisions.size());
	setting_exclude_collisions[p_collision] = NodePath(); // Reset first.
	if (is_inside_tree()) {
		Node *node = get_node_or_null(p_node_path);
		if (!node) {
			_make_collisions_dirty();
			return;
		}
		node = node->get_parent();
		if (!node || node != this) {
			_make_collisions_dirty();
			ERR_FAIL_EDMSG("Collision must be child of current SpringBoneSimulator3D.");
		}
	}
	setting_exclude_collisions[p_collision] = p_node_path;
	_make_collisions_dirty();
}

NodePath SpringBoneSimulator3D::get_exclude_collision_path(int p_index, int p_collision) const {
	ERR_FAIL_INDEX_V(p_index, (int)settings.size(), NodePath());
	const LocalVector<NodePath> &setting_exclude_collisions = settings[p_index]->exclude_collisions;
	ERR_FAIL_INDEX_V(p_collision, (int)setting_exclude_collisions.size(), NodePath());
	return setting_exclude_collisions[p_collision];
}

void SpringBoneSimulator3D::set_exclude_collision_count(int p_index, int p_count) {
	ERR_FAIL_INDEX(p_index, (int)settings.size());
	if (!are_all_child_collisions_enabled(p_index)) {
		return; // Exclude collision list is disabled.
	}
	LocalVector<NodePath> &setting_exclude_collisions = settings[p_index]->exclude_collisions;
	setting_exclude_collisions.resize(p_count);
	_make_collisions_dirty();
	notify_property_list_changed();
}

int SpringBoneSimulator3D::get_exclude_collision_count(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, (int)settings.size(), 0);
	const LocalVector<NodePath> &setting_exclude_collisions = settings[p_index]->exclude_collisions;
	return setting_exclude_collisions.size();
}

void SpringBoneSimulator3D::clear_exclude_collisions(int p_index) {
	ERR_FAIL_INDEX(p_index, (int)settings.size());
	if (!are_all_child_collisions_enabled(p_index)) {
		return; // Exclude collision list is disabled.
	}
	set_exclude_collision_count(p_index, 0);
}

void SpringBoneSimulator3D::set_collision_path(int p_index, int p_collision, const NodePath &p_node_path) {
	ERR_FAIL_INDEX(p_index, (int)settings.size());
	if (are_all_child_collisions_enabled(p_index)) {
		return; // Collision list is disabled.
	}
	LocalVector<NodePath> &setting_collisions = settings[p_index]->collisions;
	ERR_FAIL_INDEX(p_collision, (int)setting_collisions.size());
	setting_collisions[p_collision] = NodePath(); // Reset first.
	if (is_inside_tree()) {
		Node *node = get_node_or_null(p_node_path);
		if (!node) {
			_make_collisions_dirty();
			return;
		}
		node = node->get_parent();
		if (!node || node != this) {
			_make_collisions_dirty();
			ERR_FAIL_EDMSG("Collision must be child of current SpringBoneSimulator3D.");
		}
	}
	setting_collisions[p_collision] = p_node_path;
	_make_collisions_dirty();
}

NodePath SpringBoneSimulator3D::get_collision_path(int p_index, int p_collision) const {
	ERR_FAIL_INDEX_V(p_index, (int)settings.size(), NodePath());
	const LocalVector<NodePath> &setting_collisions = settings[p_index]->collisions;
	ERR_FAIL_INDEX_V(p_collision, (int)setting_collisions.size(), NodePath());
	return setting_collisions[p_collision];
}

void SpringBoneSimulator3D::set_collision_count(int p_index, int p_count) {
	ERR_FAIL_INDEX(p_index, (int)settings.size());
	if (are_all_child_collisions_enabled(p_index)) {
		return; // Collision list is disabled.
	}
	LocalVector<NodePath> &setting_collisions = settings[p_index]->collisions;
	setting_collisions.resize(p_count);
	_make_collisions_dirty();
	notify_property_list_changed();
}

int SpringBoneSimulator3D::get_collision_count(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, (int)settings.size(), 0);
	const LocalVector<NodePath> &setting_collisions = settings[p_index]->collisions;
	return setting_collisions.size();
}

void SpringBoneSimulator3D::clear_collisions(int p_index) {
	ERR_FAIL_INDEX(p_index, (int)settings.size());
	if (are_all_child_collisions_enabled(p_index)) {
		return; // Collision list is disabled.
	}
	set_collision_count(p_index, 0);
}

LocalVector<ObjectID> SpringBoneSimulator3D::get_valid_collision_instance_ids(int p_index) {
	ERR_FAIL_INDEX_V(p_index, (int)settings.size(), LocalVector<ObjectID>());
	if (collisions_dirty) {
		_find_collisions();
	}
	return settings[p_index]->cached_collisions;
}

void SpringBoneSimulator3D::set_external_force(const Vector3 &p_force) {
	external_force = p_force;
}

Vector3 SpringBoneSimulator3D::get_external_force() const {
	return external_force;
}

void SpringBoneSimulator3D::set_mutable_bone_axes(bool p_enabled) {
	mutable_bone_axes = p_enabled;
	for (SpringBone3DSetting *setting : settings) {
		setting->simulation_dirty = true;
	}
}

bool SpringBoneSimulator3D::are_bone_axes_mutable() const {
	return mutable_bone_axes;
}

void SpringBoneSimulator3D::_bind_methods() {
	// Setting.
	ClassDB::bind_method(D_METHOD("set_root_bone_name", "index", "bone_name"), &SpringBoneSimulator3D::set_root_bone_name);
	ClassDB::bind_method(D_METHOD("get_root_bone_name", "index"), &SpringBoneSimulator3D::get_root_bone_name);
	ClassDB::bind_method(D_METHOD("set_root_bone", "index", "bone"), &SpringBoneSimulator3D::set_root_bone);
	ClassDB::bind_method(D_METHOD("get_root_bone", "index"), &SpringBoneSimulator3D::get_root_bone);

	ClassDB::bind_method(D_METHOD("set_end_bone_name", "index", "bone_name"), &SpringBoneSimulator3D::set_end_bone_name);
	ClassDB::bind_method(D_METHOD("get_end_bone_name", "index"), &SpringBoneSimulator3D::get_end_bone_name);
	ClassDB::bind_method(D_METHOD("set_end_bone", "index", "bone"), &SpringBoneSimulator3D::set_end_bone);
	ClassDB::bind_method(D_METHOD("get_end_bone", "index"), &SpringBoneSimulator3D::get_end_bone);

	ClassDB::bind_method(D_METHOD("set_extend_end_bone", "index", "enabled"), &SpringBoneSimulator3D::set_extend_end_bone);
	ClassDB::bind_method(D_METHOD("is_end_bone_extended", "index"), &SpringBoneSimulator3D::is_end_bone_extended);
	ClassDB::bind_method(D_METHOD("set_end_bone_direction", "index", "bone_direction"), &SpringBoneSimulator3D::set_end_bone_direction);
	ClassDB::bind_method(D_METHOD("get_end_bone_direction", "index"), &SpringBoneSimulator3D::get_end_bone_direction);
	ClassDB::bind_method(D_METHOD("set_end_bone_length", "index", "length"), &SpringBoneSimulator3D::set_end_bone_length);
	ClassDB::bind_method(D_METHOD("get_end_bone_length", "index"), &SpringBoneSimulator3D::get_end_bone_length);

	ClassDB::bind_method(D_METHOD("set_center_from", "index", "center_from"), &SpringBoneSimulator3D::set_center_from);
	ClassDB::bind_method(D_METHOD("get_center_from", "index"), &SpringBoneSimulator3D::get_center_from);
	ClassDB::bind_method(D_METHOD("set_center_node", "index", "node_path"), &SpringBoneSimulator3D::set_center_node);
	ClassDB::bind_method(D_METHOD("get_center_node", "index"), &SpringBoneSimulator3D::get_center_node);
	ClassDB::bind_method(D_METHOD("set_center_bone_name", "index", "bone_name"), &SpringBoneSimulator3D::set_center_bone_name);
	ClassDB::bind_method(D_METHOD("get_center_bone_name", "index"), &SpringBoneSimulator3D::get_center_bone_name);
	ClassDB::bind_method(D_METHOD("set_center_bone", "index", "bone"), &SpringBoneSimulator3D::set_center_bone);
	ClassDB::bind_method(D_METHOD("get_center_bone", "index"), &SpringBoneSimulator3D::get_center_bone);

	ClassDB::bind_method(D_METHOD("set_radius", "index", "radius"), &SpringBoneSimulator3D::set_radius);
	ClassDB::bind_method(D_METHOD("get_radius", "index"), &SpringBoneSimulator3D::get_radius);
	ClassDB::bind_method(D_METHOD("set_rotation_axis", "index", "axis"), &SpringBoneSimulator3D::set_rotation_axis);
	ClassDB::bind_method(D_METHOD("get_rotation_axis", "index"), &SpringBoneSimulator3D::get_rotation_axis);
	ClassDB::bind_method(D_METHOD("set_rotation_axis_vector", "index", "vector"), &SpringBoneSimulator3D::set_rotation_axis_vector);
	ClassDB::bind_method(D_METHOD("get_rotation_axis_vector", "index"), &SpringBoneSimulator3D::get_rotation_axis_vector);
	ClassDB::bind_method(D_METHOD("set_radius_damping_curve", "index", "curve"), &SpringBoneSimulator3D::set_radius_damping_curve);
	ClassDB::bind_method(D_METHOD("get_radius_damping_curve", "index"), &SpringBoneSimulator3D::get_radius_damping_curve);
	ClassDB::bind_method(D_METHOD("set_stiffness", "index", "stiffness"), &SpringBoneSimulator3D::set_stiffness);
	ClassDB::bind_method(D_METHOD("get_stiffness", "index"), &SpringBoneSimulator3D::get_stiffness);
	ClassDB::bind_method(D_METHOD("set_stiffness_damping_curve", "index", "curve"), &SpringBoneSimulator3D::set_stiffness_damping_curve);
	ClassDB::bind_method(D_METHOD("get_stiffness_damping_curve", "index"), &SpringBoneSimulator3D::get_stiffness_damping_curve);
	ClassDB::bind_method(D_METHOD("set_drag", "index", "drag"), &SpringBoneSimulator3D::set_drag);
	ClassDB::bind_method(D_METHOD("get_drag", "index"), &SpringBoneSimulator3D::get_drag);
	ClassDB::bind_method(D_METHOD("set_drag_damping_curve", "index", "curve"), &SpringBoneSimulator3D::set_drag_damping_curve);
	ClassDB::bind_method(D_METHOD("get_drag_damping_curve", "index"), &SpringBoneSimulator3D::get_drag_damping_curve);
	ClassDB::bind_method(D_METHOD("set_gravity", "index", "gravity"), &SpringBoneSimulator3D::set_gravity);
	ClassDB::bind_method(D_METHOD("get_gravity", "index"), &SpringBoneSimulator3D::get_gravity);
	ClassDB::bind_method(D_METHOD("set_gravity_damping_curve", "index", "curve"), &SpringBoneSimulator3D::set_gravity_damping_curve);
	ClassDB::bind_method(D_METHOD("get_gravity_damping_curve", "index"), &SpringBoneSimulator3D::get_gravity_damping_curve);
	ClassDB::bind_method(D_METHOD("set_gravity_direction", "index", "gravity_direction"), &SpringBoneSimulator3D::set_gravity_direction);
	ClassDB::bind_method(D_METHOD("get_gravity_direction", "index"), &SpringBoneSimulator3D::get_gravity_direction);

	ClassDB::bind_method(D_METHOD("set_setting_count", "count"), &SpringBoneSimulator3D::set_setting_count);
	ClassDB::bind_method(D_METHOD("get_setting_count"), &SpringBoneSimulator3D::get_setting_count);
	ClassDB::bind_method(D_METHOD("clear_settings"), &SpringBoneSimulator3D::clear_settings);

	// Individual joints.
	ClassDB::bind_method(D_METHOD("set_individual_config", "index", "enabled"), &SpringBoneSimulator3D::set_individual_config);
	ClassDB::bind_method(D_METHOD("is_config_individual", "index"), &SpringBoneSimulator3D::is_config_individual);

	ClassDB::bind_method(D_METHOD("get_joint_bone_name", "index", "joint"), &SpringBoneSimulator3D::get_joint_bone_name);
	ClassDB::bind_method(D_METHOD("get_joint_bone", "index", "joint"), &SpringBoneSimulator3D::get_joint_bone);
	ClassDB::bind_method(D_METHOD("set_joint_rotation_axis", "index", "joint", "axis"), &SpringBoneSimulator3D::set_joint_rotation_axis);
	ClassDB::bind_method(D_METHOD("get_joint_rotation_axis", "index", "joint"), &SpringBoneSimulator3D::get_joint_rotation_axis);
	ClassDB::bind_method(D_METHOD("set_joint_rotation_axis_vector", "index", "joint", "vector"), &SpringBoneSimulator3D::set_joint_rotation_axis_vector);
	ClassDB::bind_method(D_METHOD("get_joint_rotation_axis_vector", "index", "joint"), &SpringBoneSimulator3D::get_joint_rotation_axis_vector);
	ClassDB::bind_method(D_METHOD("set_joint_radius", "index", "joint", "radius"), &SpringBoneSimulator3D::set_joint_radius);
	ClassDB::bind_method(D_METHOD("get_joint_radius", "index", "joint"), &SpringBoneSimulator3D::get_joint_radius);
	ClassDB::bind_method(D_METHOD("set_joint_stiffness", "index", "joint", "stiffness"), &SpringBoneSimulator3D::set_joint_stiffness);
	ClassDB::bind_method(D_METHOD("get_joint_stiffness", "index", "joint"), &SpringBoneSimulator3D::get_joint_stiffness);
	ClassDB::bind_method(D_METHOD("set_joint_drag", "index", "joint", "drag"), &SpringBoneSimulator3D::set_joint_drag);
	ClassDB::bind_method(D_METHOD("get_joint_drag", "index", "joint"), &SpringBoneSimulator3D::get_joint_drag);
	ClassDB::bind_method(D_METHOD("set_joint_gravity", "index", "joint", "gravity"), &SpringBoneSimulator3D::set_joint_gravity);
	ClassDB::bind_method(D_METHOD("get_joint_gravity", "index", "joint"), &SpringBoneSimulator3D::get_joint_gravity);
	ClassDB::bind_method(D_METHOD("set_joint_gravity_direction", "index", "joint", "gravity_direction"), &SpringBoneSimulator3D::set_joint_gravity_direction);
	ClassDB::bind_method(D_METHOD("get_joint_gravity_direction", "index", "joint"), &SpringBoneSimulator3D::get_joint_gravity_direction);

	ClassDB::bind_method(D_METHOD("get_joint_count", "index"), &SpringBoneSimulator3D::get_joint_count);

	// Individual collisions.
	ClassDB::bind_method(D_METHOD("set_enable_all_child_collisions", "index", "enabled"), &SpringBoneSimulator3D::set_enable_all_child_collisions);
	ClassDB::bind_method(D_METHOD("are_all_child_collisions_enabled", "index"), &SpringBoneSimulator3D::are_all_child_collisions_enabled);

	ClassDB::bind_method(D_METHOD("set_exclude_collision_path", "index", "collision", "node_path"), &SpringBoneSimulator3D::set_exclude_collision_path);
	ClassDB::bind_method(D_METHOD("get_exclude_collision_path", "index", "collision"), &SpringBoneSimulator3D::get_exclude_collision_path);

	ClassDB::bind_method(D_METHOD("set_exclude_collision_count", "index", "count"), &SpringBoneSimulator3D::set_exclude_collision_count);
	ClassDB::bind_method(D_METHOD("get_exclude_collision_count", "index"), &SpringBoneSimulator3D::get_exclude_collision_count);
	ClassDB::bind_method(D_METHOD("clear_exclude_collisions", "index"), &SpringBoneSimulator3D::clear_exclude_collisions);

	ClassDB::bind_method(D_METHOD("set_collision_path", "index", "collision", "node_path"), &SpringBoneSimulator3D::set_collision_path);
	ClassDB::bind_method(D_METHOD("get_collision_path", "index", "collision"), &SpringBoneSimulator3D::get_collision_path);

	ClassDB::bind_method(D_METHOD("set_collision_count", "index", "count"), &SpringBoneSimulator3D::set_collision_count);
	ClassDB::bind_method(D_METHOD("get_collision_count", "index"), &SpringBoneSimulator3D::get_collision_count);
	ClassDB::bind_method(D_METHOD("clear_collisions", "index"), &SpringBoneSimulator3D::clear_collisions);

	ClassDB::bind_method(D_METHOD("set_external_force", "force"), &SpringBoneSimulator3D::set_external_force);
	ClassDB::bind_method(D_METHOD("get_external_force"), &SpringBoneSimulator3D::get_external_force);

	ClassDB::bind_method(D_METHOD("set_mutable_bone_axes", "enabled"), &SpringBoneSimulator3D::set_mutable_bone_axes);
	ClassDB::bind_method(D_METHOD("are_bone_axes_mutable"), &SpringBoneSimulator3D::are_bone_axes_mutable);

	// To process manually.
	ClassDB::bind_method(D_METHOD("reset"), &SpringBoneSimulator3D::reset);

	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "external_force", PROPERTY_HINT_RANGE, "-99999,99999,or_greater,or_less,hide_control,suffix:m/s"), "set_external_force", "get_external_force");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "mutable_bone_axes"), "set_mutable_bone_axes", "are_bone_axes_mutable");
	ADD_ARRAY_COUNT("Settings", "setting_count", "set_setting_count", "get_setting_count", "settings/");

	BIND_ENUM_CONSTANT(CENTER_FROM_WORLD_ORIGIN);
	BIND_ENUM_CONSTANT(CENTER_FROM_NODE);
	BIND_ENUM_CONSTANT(CENTER_FROM_BONE);
}

void SpringBoneSimulator3D::_skeleton_changed(Skeleton3D *p_old, Skeleton3D *p_new) {
	if (p_old && p_old->is_connected(SNAME("rest_updated"), callable_mp(this, &SpringBoneSimulator3D::_make_all_joints_dirty))) {
		p_old->disconnect(SNAME("rest_updated"), callable_mp(this, &SpringBoneSimulator3D::_make_all_joints_dirty));
	}
	if (p_new && !p_new->is_connected(SNAME("rest_updated"), callable_mp(this, &SpringBoneSimulator3D::_make_all_joints_dirty))) {
		p_new->connect(SNAME("rest_updated"), callable_mp(this, &SpringBoneSimulator3D::_make_all_joints_dirty));
	}
	_make_all_joints_dirty();
}

void SpringBoneSimulator3D::_validate_bone_names() {
	for (uint32_t i = 0; i < settings.size(); i++) {
		// Prior bone name.
		if (!settings[i]->root_bone_name.is_empty()) {
			set_root_bone_name(i, settings[i]->root_bone_name);
		} else if (settings[i]->root_bone != -1) {
			set_root_bone(i, settings[i]->root_bone);
		}
		// Prior bone name.
		if (!settings[i]->end_bone_name.is_empty()) {
			set_end_bone_name(i, settings[i]->end_bone_name);
		} else if (settings[i]->end_bone != -1) {
			set_end_bone(i, settings[i]->end_bone);
		}
	}
}

void SpringBoneSimulator3D::_make_joints_dirty(int p_index, bool p_reset) {
	ERR_FAIL_INDEX(p_index, (int)settings.size());
	settings[p_index]->joints_dirty = true;
	if (joints_dirty) {
		return;
	}
	joints_dirty = true;
	callable_mp(this, &SpringBoneSimulator3D::_update_joints).call_deferred(p_reset);
}

void SpringBoneSimulator3D::_make_all_joints_dirty() {
	for (uint32_t i = 0; i < settings.size(); i++) {
		_update_joint_array(i);
	}
}

void SpringBoneSimulator3D::add_child_notify(Node *p_child) {
	if (Object::cast_to<SpringBoneCollision3D>(p_child)) {
		_make_collisions_dirty();
	}
}

void SpringBoneSimulator3D::move_child_notify(Node *p_child) {
	if (Object::cast_to<SpringBoneCollision3D>(p_child)) {
		_make_collisions_dirty();
	}
}

void SpringBoneSimulator3D::remove_child_notify(Node *p_child) {
	if (Object::cast_to<SpringBoneCollision3D>(p_child)) {
		_make_collisions_dirty();
	}
}

void SpringBoneSimulator3D::_validate_rotation_axes(Skeleton3D *p_skeleton) const {
	for (uint32_t i = 0; i < settings.size(); i++) {
		for (uint32_t j = 0; j < settings[i]->joints.size(); j++) {
			_validate_rotation_axis(p_skeleton, i, j);
		}
	}
}

void SpringBoneSimulator3D::_validate_rotation_axis(Skeleton3D *p_skeleton, int p_index, int p_joint) const {
	RotationAxis axis = settings[p_index]->joints[p_joint]->rotation_axis;
	if (axis == ROTATION_AXIS_ALL) {
		return;
	}
	Vector3 rot = get_joint_rotation_axis_vector(p_index, p_joint).normalized();
	Vector3 fwd;
	if (p_joint < (int)settings[p_index]->joints.size() - 1) {
		fwd = p_skeleton->get_bone_rest(settings[p_index]->joints[p_joint + 1]->bone).origin;
	} else if (settings[p_index]->extend_end_bone) {
		fwd = get_end_bone_axis(settings[p_index]->end_bone, settings[p_index]->end_bone_direction);
		if (fwd.is_zero_approx()) {
			return;
		}
	}
	fwd.normalize();
	if (Math::is_equal_approx(Math::abs(rot.dot(fwd)), 1)) {
		WARN_PRINT_ED("Setting: " + itos(p_index) + " Joint: " + itos(p_joint) + ": Rotation axis and forward vector are colinear. This is not advised as it may cause unwanted rotation.");
	}
}

void SpringBoneSimulator3D::_find_collisions() {
	if (!collisions_dirty) {
		return;
	}
	collisions.clear();
	for (int i = 0; i < get_child_count(); i++) {
		SpringBoneCollision3D *c = Object::cast_to<SpringBoneCollision3D>(get_child(i));
		if (c) {
			collisions.push_back(c->get_instance_id());
		}
	}

	bool setting_updated = false;

	for (uint32_t i = 0; i < settings.size(); i++) {
		LocalVector<ObjectID> &cache = settings[i]->cached_collisions;
		cache.clear();
		if (!settings[i]->enable_all_child_collisions) {
			// Allow list.
			LocalVector<NodePath> &setting_collisions = settings[i]->collisions;
			for (uint32_t j = 0; j < setting_collisions.size(); j++) {
				Node *n = get_node_or_null(setting_collisions[j]);
				if (!n) {
					continue;
				}
				ObjectID id = n->get_instance_id();
				if (!collisions.has(id)) {
					setting_collisions[j] = NodePath(); // Clear path if not found.
				} else {
					cache.push_back(id);
				}
			}
		} else {
			// Deny list.
			LocalVector<uint32_t> masks;
			LocalVector<NodePath> &setting_exclude_collisions = settings[i]->exclude_collisions;
			for (uint32_t j = 0; j < setting_exclude_collisions.size(); j++) {
				Node *n = get_node_or_null(setting_exclude_collisions[j]);
				if (!n) {
					continue;
				}
				ObjectID id = n->get_instance_id();
				int find = collisions.find(id);
				if (find < 0) {
					setting_exclude_collisions[j] = NodePath(); // Clear path if not found.
				} else {
					masks.push_back((uint32_t)find);
				}
			}
			uint32_t mask_index = 0;
			for (uint32_t j = 0; j < collisions.size(); j++) {
				if (mask_index < masks.size() && j == masks[mask_index]) {
					mask_index++;
					continue;
				}
				cache.push_back(collisions[j]);
			}
		}
	}

	collisions_dirty = false;

	if (setting_updated) {
		notify_property_list_changed();
	}
}

void SpringBoneSimulator3D::_process_collisions() {
	for (const ObjectID &oid : collisions) {
		Object *t_obj = ObjectDB::get_instance(oid);
		if (!t_obj) {
			continue;
		}
		SpringBoneCollision3D *col = Object::cast_to<SpringBoneCollision3D>(t_obj);
		if (!col) {
			continue;
		}
		col->sync_pose();
	}
}

void SpringBoneSimulator3D::_make_collisions_dirty() {
	collisions_dirty = true;
}

void SpringBoneSimulator3D::_update_joint_array(int p_index) {
	_make_joints_dirty(p_index, true);

	Skeleton3D *sk = get_skeleton();
	int current_bone = settings[p_index]->end_bone;
	int root_bone = settings[p_index]->root_bone;
	if (!sk || current_bone < 0 || root_bone < 0) {
		set_joint_count(p_index, 0);
		return;
	}

	// Validation.
	bool valid = false;
	while (current_bone >= 0) {
		if (current_bone == root_bone) {
			valid = true;
			break;
		}
		current_bone = sk->get_bone_parent(current_bone);
	}

	if (!valid) {
		set_joint_count(p_index, 0);
		ERR_FAIL_EDMSG("End bone must be the same as or a child of the root bone.");
	}

	LocalVector<int> new_joints;
	current_bone = settings[p_index]->end_bone;
	while (current_bone != root_bone) {
		new_joints.push_back(current_bone);
		current_bone = sk->get_bone_parent(current_bone);
	}
	new_joints.push_back(current_bone);
	new_joints.reverse();

	set_joint_count(p_index, new_joints.size());
	for (uint32_t i = 0; i < new_joints.size(); i++) {
		_set_joint_bone(p_index, i, new_joints[i]);
	}
}

void SpringBoneSimulator3D::_update_joints(bool p_reset) {
	if (!joints_dirty) {
		return;
	}
	for (uint32_t i = 0; i < settings.size(); i++) {
		if (!settings[i]->joints_dirty) {
			continue;
		}
		if (settings[i]->individual_config) {
			settings[i]->simulation_dirty = p_reset;
			settings[i]->joints_dirty = false;
			continue; // Abort.
		}
		LocalVector<SpringBone3DJointSetting *> &joints = settings[i]->joints;
		float unit = joints.size() > 0 ? (1.0 / float(joints.size() - 1)) : 0.0;
		for (uint32_t j = 0; j < joints.size(); j++) {
			float offset = j * unit;

			if (settings[i]->radius_damping_curve.is_valid()) {
				joints[j]->radius = settings[i]->radius * settings[i]->radius_damping_curve->sample_baked(offset);
			} else {
				joints[j]->radius = settings[i]->radius;
			}

			if (settings[i]->stiffness_damping_curve.is_valid()) {
				joints[j]->stiffness = settings[i]->stiffness * settings[i]->stiffness_damping_curve->sample_baked(offset);
			} else {
				joints[j]->stiffness = settings[i]->stiffness;
			}

			if (settings[i]->drag_damping_curve.is_valid()) {
				joints[j]->drag = settings[i]->drag * settings[i]->drag_damping_curve->sample_baked(offset);
			} else {
				joints[j]->drag = settings[i]->drag;
			}

			if (settings[i]->gravity_damping_curve.is_valid()) {
				joints[j]->gravity = settings[i]->gravity * settings[i]->gravity_damping_curve->sample_baked(offset);
			} else {
				joints[j]->gravity = settings[i]->gravity;
			}

			joints[j]->gravity_direction = settings[i]->gravity_direction;
			joints[j]->rotation_axis = settings[i]->rotation_axis;
			joints[j]->rotation_axis_vector = settings[i]->rotation_axis_vector;
		}
		settings[i]->simulation_dirty = p_reset;
		settings[i]->joints_dirty = false;
	}
	joints_dirty = false;
	Skeleton3D *sk = get_skeleton();
	if (sk) {
		_validate_rotation_axes(sk);
	}
#ifdef TOOLS_ENABLED
	_make_gizmo_dirty();
#endif // TOOLS_ENABLED
}

void SpringBoneSimulator3D::_update_bone_axis(Skeleton3D *p_skeleton, SpringBone3DSetting *p_setting) {
#ifdef TOOLS_ENABLED
	bool changed = false;
#endif // TOOLS_ENABLED
	const LocalVector<SpringBone3DJointSetting *> &joints = p_setting->joints;
	int len = (int)joints.size() - 1;
	for (int j = 0; j < len; j++) {
		if (!joints[j]->verlet) {
			continue;
		}
		Vector3 axis = p_skeleton->get_bone_pose(joints[j + 1]->bone).origin;
		if (axis.is_zero_approx()) {
			continue;
		}
		// Less computing.
#ifdef TOOLS_ENABLED
		if (!changed) {
			Vector3 old_v = joints[j]->verlet->forward_vector;
			joints[j]->verlet->forward_vector = snap_vector_to_plane(joints[j]->get_rotation_axis_vector(), axis.normalized());
			changed = changed || !old_v.is_equal_approx(joints[j]->verlet->forward_vector);
			float old_l = joints[j]->verlet->length;
			joints[j]->verlet->length = axis.length();
			changed = changed || !Math::is_equal_approx(old_l, joints[j]->verlet->length);
		} else {
			joints[j]->verlet->forward_vector = snap_vector_to_plane(joints[j]->get_rotation_axis_vector(), axis.normalized());
			joints[j]->verlet->length = axis.length();
		}
#else
		joints[j]->verlet->forward_vector = snap_vector_to_plane(joints[j]->get_rotation_axis_vector(), axis.normalized());
		joints[j]->verlet->length = axis.length();
#endif // TOOLS_ENABLED
	}
	if (p_setting->extend_end_bone && len >= 0) {
		if (joints[len]->verlet) {
			Vector3 axis = get_end_bone_axis(p_setting->end_bone, p_setting->end_bone_direction);
			if (!axis.is_zero_approx()) {
				joints[len]->verlet->forward_vector = snap_vector_to_plane(joints[len]->get_rotation_axis_vector(), axis.normalized());
				joints[len]->verlet->length = p_setting->end_bone_length;
			}
		}
	}
#ifdef TOOLS_ENABLED
	if (changed) {
		_make_gizmo_dirty();
	}
#endif // TOOLS_ENABLED
}

#ifdef TOOLS_ENABLED
Vector3 SpringBoneSimulator3D::get_bone_vector(int p_index, int p_joint) const {
	Skeleton3D *skeleton = get_skeleton();
	if (!skeleton) {
		return Vector3();
	}
	ERR_FAIL_INDEX_V(p_index, (int)settings.size(), Vector3());
	SpringBone3DSetting *setting = settings[p_index];
	if (!setting) {
		return Vector3();
	}
	const LocalVector<SpringBone3DJointSetting *> &joints = setting->joints;
	ERR_FAIL_INDEX_V(p_joint, (int)joints.size(), Vector3());
	if (!joints[p_joint]->verlet) {
		if (p_joint == (int)joints.size() - 1) {
			return get_end_bone_axis(setting->end_bone, setting->end_bone_direction) * setting->end_bone_length;
		}
		return mutable_bone_axes ? skeleton->get_bone_pose(joints[p_joint + 1]->bone).origin : skeleton->get_bone_rest(joints[p_joint + 1]->bone).origin;
	}
	return joints[p_joint]->verlet->forward_vector * joints[p_joint]->verlet->length;
}

void SpringBoneSimulator3D::_make_gizmo_dirty() {
	if (gizmo_dirty) {
		return;
	}
	gizmo_dirty = true;
	callable_mp(this, &SpringBoneSimulator3D::_redraw_gizmo).call_deferred();
}

void SpringBoneSimulator3D::_redraw_gizmo() {
	update_gizmos();
	gizmo_dirty = false;
}
#endif

void SpringBoneSimulator3D::_set_active(bool p_active) {
	if (p_active) {
		reset();
	}
}

void SpringBoneSimulator3D::_process_modification(double p_delta) {
	Skeleton3D *skeleton = get_skeleton();
	if (!skeleton) {
		return;
	}
	_find_collisions();
	_process_collisions();

#ifdef TOOLS_ENABLED
	if (saving) {
		return; // Collision position has been reset but we don't want to process simulating on saving. Abort.
	}
#endif // TOOLS_ENABLED

	for (uint32_t i = 0; i < settings.size(); i++) {
		_init_joints(skeleton, settings[i]);
		_process_joints(p_delta, skeleton, settings[i]->joints, get_valid_collision_instance_ids(i), settings[i]->cached_center, settings[i]->cached_inverted_center, settings[i]->cached_inverted_center.basis.get_rotation_quaternion());
	}
}

void SpringBoneSimulator3D::reset() {
	if (!is_inside_tree()) {
		return;
	}
	Skeleton3D *skeleton = get_skeleton();
	if (!skeleton) {
		return;
	}
	_find_collisions();
	_process_collisions();
	for (uint32_t i = 0; i < settings.size(); i++) {
		_make_joints_dirty(i, true);
		_init_joints(skeleton, settings[i]);
	}
}

void SpringBoneSimulator3D::_init_joints(Skeleton3D *p_skeleton, SpringBone3DSetting *setting) {
	if (setting->center_from == CENTER_FROM_WORLD_ORIGIN) {
		setting->cached_center = p_skeleton->get_global_transform_interpolated();
	} else if (setting->center_from == CENTER_FROM_NODE) {
		if (setting->center_node == NodePath()) {
			setting->cached_center = Transform3D();
		} else {
			Node3D *nd = Object::cast_to<Node3D>(get_node_or_null(setting->center_node));
			if (!nd) {
				setting->cached_center = Transform3D();
			} else {
				setting->cached_center = nd->get_global_transform_interpolated().affine_inverse() * p_skeleton->get_global_transform_interpolated();
			}
		}
	} else {
		if (setting->center_bone >= 0) {
			setting->cached_center = p_skeleton->get_bone_global_pose(setting->center_bone);
		} else {
			setting->cached_center = Transform3D();
		}
	}
	setting->cached_inverted_center = setting->cached_center.affine_inverse();

	if (!setting->simulation_dirty) {
		if (mutable_bone_axes) {
			_update_bone_axis(p_skeleton, setting);
		}
		return;
	}
	for (uint32_t i = 0; i < setting->joints.size(); i++) {
		if (setting->joints[i]->verlet) {
			memdelete(setting->joints[i]->verlet);
			setting->joints[i]->verlet = nullptr;
		}
		if (i < setting->joints.size() - 1) {
			Vector3 axis = p_skeleton->get_bone_rest(setting->joints[i + 1]->bone).origin;
			if (axis.is_zero_approx()) {
				continue;
			}
			setting->joints[i]->verlet = memnew(SpringBone3DVerletInfo);
			setting->joints[i]->verlet->current_tail = setting->cached_center.xform(p_skeleton->get_bone_global_pose(setting->joints[i]->bone).xform(axis));
			setting->joints[i]->verlet->prev_tail = setting->joints[i]->verlet->current_tail;
			setting->joints[i]->verlet->forward_vector = snap_vector_to_plane(setting->joints[i]->get_rotation_axis_vector(), axis.normalized());
			setting->joints[i]->verlet->length = axis.length();
			setting->joints[i]->verlet->current_rot = Quaternion(0, 0, 0, 1);
		} else if (setting->extend_end_bone && setting->end_bone_length > 0) {
			Vector3 axis = get_end_bone_axis(setting->end_bone, setting->end_bone_direction);
			if (axis.is_zero_approx()) {
				continue;
			}
			setting->joints[i]->verlet = memnew(SpringBone3DVerletInfo);
			setting->joints[i]->verlet->forward_vector = snap_vector_to_plane(setting->joints[i]->get_rotation_axis_vector(), axis.normalized());
			setting->joints[i]->verlet->length = setting->end_bone_length;
			setting->joints[i]->verlet->current_tail = setting->cached_center.xform(p_skeleton->get_bone_global_pose(setting->joints[i]->bone).xform(axis * setting->end_bone_length));
			setting->joints[i]->verlet->prev_tail = setting->joints[i]->verlet->current_tail;
			setting->joints[i]->verlet->current_rot = Quaternion(0, 0, 0, 1);
		}
	}
	if (mutable_bone_axes) {
		_update_bone_axis(p_skeleton, setting);
#ifdef TOOLS_ENABLED
	} else {
		_make_gizmo_dirty();
#endif // TOOLS_ENABLED
	}
	setting->simulation_dirty = false;
}

void SpringBoneSimulator3D::_process_joints(double p_delta, Skeleton3D *p_skeleton, LocalVector<SpringBone3DJointSetting *> &p_joints, const LocalVector<ObjectID> &p_collisions, const Transform3D &p_center_transform, const Transform3D &p_inverted_center_transform, const Quaternion &p_inverted_center_rotation) {
	for (uint32_t i = 0; i < p_joints.size(); i++) {
		SpringBone3DVerletInfo *verlet = p_joints[i]->verlet;
		if (!verlet) {
			continue; // Means not extended end bone.
		}
		Transform3D current_global_pose = p_skeleton->get_bone_global_pose(p_joints[i]->bone);
		Transform3D current_world_pose = p_center_transform * current_global_pose;
		Quaternion current_rot = current_global_pose.basis.get_rotation_quaternion();
		Vector3 current_origin = p_center_transform.xform(current_global_pose.origin);
		Vector3 external = p_inverted_center_rotation.xform((external_force + p_joints[i]->gravity_direction * p_joints[i]->gravity) * p_delta);

		// Integration of velocity by verlet.
		Vector3 next_tail = verlet->current_tail +
				(verlet->current_tail - verlet->prev_tail) * (1.0 - p_joints[i]->drag) +
				p_center_transform.basis.get_rotation_quaternion().xform(current_rot.xform(verlet->forward_vector * (p_joints[i]->stiffness * p_delta)) + external);
		// Snap to plane if axis locked.
		if (p_joints[i]->rotation_axis != ROTATION_AXIS_ALL) {
			next_tail = current_world_pose.origin + current_world_pose.basis.get_rotation_quaternion().xform(snap_vector_to_plane(p_joints[i]->get_rotation_axis_vector(), current_world_pose.basis.get_rotation_quaternion().xform_inv(next_tail - current_world_pose.origin)));
		}
		// Limit bone length.
		next_tail = limit_length(current_origin, next_tail, verlet->length);

		// Collision movement.
		for (uint32_t j = 0; j < p_collisions.size(); j++) {
			Object *obj = ObjectDB::get_instance(p_collisions[j]);
			if (!obj) {
				continue;
			}
			SpringBoneCollision3D *col = Object::cast_to<SpringBoneCollision3D>(obj);
			if (col) {
				// Collider movement should separate from the effect of the center.
				next_tail = col->collide(p_center_transform, p_joints[i]->radius, verlet->length, next_tail);
				// Snap to plane if axis locked.
				if (p_joints[i]->rotation_axis != ROTATION_AXIS_ALL) {
					next_tail = current_world_pose.origin + current_world_pose.basis.get_rotation_quaternion().xform(snap_vector_to_plane(p_joints[i]->get_rotation_axis_vector(), current_world_pose.basis.get_rotation_quaternion().xform_inv(next_tail - current_world_pose.origin)));
				}
				// Limit bone length.
				next_tail = limit_length(current_origin, next_tail, verlet->length);
			}
		}

		// Store current tails for next process.
		verlet->prev_tail = verlet->current_tail;
		verlet->current_tail = next_tail;

		// Convert position to rotation.
		Vector3 from = current_rot.xform(verlet->forward_vector);
		Vector3 to = p_inverted_center_transform.basis.xform(next_tail - current_origin);
		from.normalize();
		to.normalize();
		Quaternion from_to = get_from_to_rotation(from, to, verlet->current_rot);
		verlet->current_rot = from_to;

		// Apply rotation.
		from_to *= current_rot;
		from_to = get_local_pose_rotation(p_skeleton, p_joints[i]->bone, from_to);
		p_skeleton->set_bone_pose_rotation(p_joints[i]->bone, from_to);
	}
}

SpringBoneSimulator3D::~SpringBoneSimulator3D() {
	clear_settings();
}
