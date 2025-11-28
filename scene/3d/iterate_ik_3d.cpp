/**************************************************************************/
/*  iterate_ik_3d.cpp                                                     */
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

#include "iterate_ik_3d.h"

bool IterateIK3D::_set(const StringName &p_path, const Variant &p_value) {
	String path = p_path;

	if (path.begins_with("settings/")) {
		int which = path.get_slicec('/', 1).to_int();
		String what = path.get_slicec('/', 2);
		ERR_FAIL_INDEX_V(which, (int)settings.size(), false);

		if (what == "target_node") {
			set_target_node(which, p_value);
		} else if (what == "joints") {
			int idx = path.get_slicec('/', 3).to_int();
			String prop = path.get_slicec('/', 4);
			if (prop == "rotation_axis") {
				set_joint_rotation_axis(which, idx, static_cast<RotationAxis>((int)p_value));
			} else if (prop == "rotation_axis_vector") {
				set_joint_rotation_axis_vector(which, idx, p_value);
			} else if (prop == "limitation") {
				String opt = path.get_slicec('/', 5);
				if (opt.is_empty()) {
					set_joint_limitation(which, idx, p_value);
				} else if (opt == "right_axis") {
					set_joint_limitation_right_axis(which, idx, p_value);
				} else if (opt == "right_axis_vector") {
					set_joint_limitation_right_axis_vector(which, idx, p_value);
				} else if (opt == "rotation_offset") {
					set_joint_limitation_rotation_offset(which, idx, p_value);
				} else {
					return false;
				}
			} else {
				return false;
			}
		} else {
			return false;
		}
	}
	return true;
}

bool IterateIK3D::_get(const StringName &p_path, Variant &r_ret) const {
	String path = p_path;

	if (path.begins_with("settings/")) {
		int which = path.get_slicec('/', 1).to_int();
		String what = path.get_slicec('/', 2);
		ERR_FAIL_INDEX_V(which, (int)settings.size(), false);

		if (what == "target_node") {
			r_ret = get_target_node(which);
		} else if (what == "joints") {
			int idx = path.get_slicec('/', 3).to_int();
			String prop = path.get_slicec('/', 4);
			if (prop == "rotation_axis") {
				r_ret = (int)get_joint_rotation_axis(which, idx);
			} else if (prop == "rotation_axis_vector") {
				r_ret = get_joint_rotation_axis_vector(which, idx);
			} else if (prop == "limitation") {
				String opt = path.get_slicec('/', 5);
				if (opt.is_empty()) {
					r_ret = get_joint_limitation(which, idx);
				} else if (opt == "right_axis") {
					r_ret = get_joint_limitation_right_axis(which, idx);
				} else if (opt == "right_axis_vector") {
					r_ret = get_joint_limitation_right_axis_vector(which, idx);
				} else if (opt == "rotation_offset") {
					r_ret = get_joint_limitation_rotation_offset(which, idx);
				} else {
					return false;
				}
			} else {
				return false;
			}
		} else {
			return false;
		}
	}
	return true;
}

void IterateIK3D::_get_property_list(List<PropertyInfo> *p_list) const {
	LocalVector<PropertyInfo> props;
	for (uint32_t i = 0; i < settings.size(); i++) {
		String path = "settings/" + itos(i) + "/";
		p_list->push_back(PropertyInfo(Variant::NODE_PATH, path + "target_node"));
		for (uint32_t j = 0; j < iterate_settings[i]->joints.size(); j++) {
			String joint_path = path + "joints/" + itos(j) + "/";
			props.push_back(PropertyInfo(Variant::INT, joint_path + "rotation_axis", PROPERTY_HINT_ENUM, SkeletonModifier3D::get_hint_rotation_axis()));
			props.push_back(PropertyInfo(Variant::VECTOR3, joint_path + "rotation_axis_vector"));
			props.push_back(PropertyInfo(Variant::OBJECT, joint_path + "limitation", PROPERTY_HINT_RESOURCE_TYPE, "JointLimitation3D"));
			props.push_back(PropertyInfo(Variant::INT, joint_path + "limitation/right_axis", PROPERTY_HINT_ENUM, SkeletonModifier3D::get_hint_secondary_direction()));
			props.push_back(PropertyInfo(Variant::VECTOR3, joint_path + "limitation/right_axis_vector"));
			props.push_back(PropertyInfo(Variant::QUATERNION, joint_path + "limitation/rotation_offset"));
		}
	}

	ChainIK3D::get_property_list(p_list);

	for (PropertyInfo &p : props) {
		_validate_dynamic_prop(p);
		p_list->push_back(p);
	}
}

void IterateIK3D::_validate_dynamic_prop(PropertyInfo &p_property) const {
	PackedStringArray split = p_property.name.split("/");
	if (split.size() > 3 && split[0] == "settings") {
		int which = split[1].to_int();
		int joint = split[3].to_int();
		// Joints option.
		if (split[2] == "joints" && split.size() > 4) {
			if (split[4] == "rotation_axis_vector" && get_joint_rotation_axis(which, joint) != ROTATION_AXIS_CUSTOM) {
				p_property.usage = PROPERTY_USAGE_NONE;
			}
			if (split[4] == "limitation" && split.size() > 5) {
				if (get_joint_limitation(which, joint).is_null()) {
					p_property.usage = PROPERTY_USAGE_NONE;
				} else if (split[5] == "right_axis_vector" && get_joint_limitation_right_axis(which, joint) != SECONDARY_DIRECTION_CUSTOM) {
					p_property.usage = PROPERTY_USAGE_NONE;
				}
			}
		}
	}
}

PackedStringArray IterateIK3D::get_configuration_warnings() const {
	PackedStringArray warnings = SkeletonModifier3D::get_configuration_warnings();
	for (uint32_t i = 0; i < iterate_settings.size(); i++) {
		if (iterate_settings[i]->target_node.is_empty()) {
			warnings.push_back(RTR("Detecting settings with no target set! IterateIK3D must have a target to work."));
			break;
		}
	}
	return warnings;
}

void IterateIK3D::set_max_iterations(int p_max_iterations) {
	max_iterations = p_max_iterations;
}

int IterateIK3D::get_max_iterations() const {
	return max_iterations;
}

void IterateIK3D::set_min_distance(double p_min_distance) {
	min_distance = p_min_distance;
}

double IterateIK3D::get_min_distance() const {
	return min_distance;
}

void IterateIK3D::set_angular_delta_limit(double p_angular_delta_limit) {
	angular_delta_limit = p_angular_delta_limit;
}

double IterateIK3D::get_angular_delta_limit() const {
	return angular_delta_limit;
}

void IterateIK3D::set_deterministic(bool p_deterministic) {
	deterministic = p_deterministic;
}

bool IterateIK3D::is_deterministic() const {
	return deterministic;
}

// Setting.

void IterateIK3D::set_target_node(int p_index, const NodePath &p_node_path) {
	ERR_FAIL_INDEX(p_index, (int)settings.size());
	iterate_settings[p_index]->target_node = p_node_path;
	update_configuration_warnings();
}

NodePath IterateIK3D::get_target_node(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, (int)settings.size(), NodePath());
	return iterate_settings[p_index]->target_node;
}

// Individual joints.

void IterateIK3D::set_joint_rotation_axis(int p_index, int p_joint, RotationAxis p_axis) {
	ERR_FAIL_INDEX(p_index, (int)settings.size());
	const LocalVector<IterateIK3DJointSetting *> &joint_settings = iterate_settings[p_index]->joint_settings;
	ERR_FAIL_INDEX(p_joint, (int)joint_settings.size());
	joint_settings[p_joint]->rotation_axis = p_axis;
	Skeleton3D *sk = get_skeleton();
	if (sk) {
		_validate_axis(sk, p_index, p_joint);
	}
	notify_property_list_changed();
	_make_simulation_dirty(p_index); // Snapping to planes is needed in the initialization, so need to restructure.
}

SkeletonModifier3D::RotationAxis IterateIK3D::get_joint_rotation_axis(int p_index, int p_joint) const {
	ERR_FAIL_INDEX_V(p_index, (int)settings.size(), ROTATION_AXIS_ALL);
	const LocalVector<IterateIK3DJointSetting *> &joint_settings = iterate_settings[p_index]->joint_settings;
	ERR_FAIL_INDEX_V(p_joint, (int)joint_settings.size(), ROTATION_AXIS_ALL);
	return joint_settings[p_joint]->rotation_axis;
}

void IterateIK3D::set_joint_rotation_axis_vector(int p_index, int p_joint, const Vector3 &p_vector) {
	ERR_FAIL_INDEX(p_index, (int)settings.size());
	const LocalVector<IterateIK3DJointSetting *> &joint_settings = iterate_settings[p_index]->joint_settings;
	ERR_FAIL_INDEX(p_joint, (int)joint_settings.size());
	joint_settings[p_joint]->rotation_axis_vector = p_vector;
	Skeleton3D *sk = get_skeleton();
	if (sk) {
		_validate_axis(sk, p_index, p_joint);
	}
	_make_simulation_dirty(p_index); // Snapping to planes is needed in the initialization, so need to restructure.
}

Vector3 IterateIK3D::get_joint_rotation_axis_vector(int p_index, int p_joint) const {
	ERR_FAIL_INDEX_V(p_index, (int)settings.size(), Vector3());
	const LocalVector<IterateIK3DJointSetting *> &joint_settings = iterate_settings[p_index]->joint_settings;
	ERR_FAIL_INDEX_V(p_joint, (int)joint_settings.size(), Vector3());
	return joint_settings[p_joint]->get_rotation_axis_vector();
}

Quaternion IterateIK3D::get_joint_limitation_space(int p_index, int p_joint, const Vector3 &p_forward) const {
	ERR_FAIL_INDEX_V(p_index, (int)settings.size(), Quaternion());
	const LocalVector<IterateIK3DJointSetting *> &joint_settings = iterate_settings[p_index]->joint_settings;
	ERR_FAIL_INDEX_V(p_joint, (int)joint_settings.size(), Quaternion());
	return joint_settings[p_joint]->get_limitation_space(p_forward);
}

void IterateIK3D::set_joint_limitation(int p_index, int p_joint, const Ref<JointLimitation3D> &p_limitation) {
	ERR_FAIL_INDEX(p_index, (int)settings.size());
	const LocalVector<IterateIK3DJointSetting *> &joint_settings = iterate_settings[p_index]->joint_settings;
	ERR_FAIL_INDEX(p_joint, (int)joint_settings.size());
	_unbind_joint_limitation(p_index, p_joint);
	joint_settings[p_joint]->limitation = p_limitation;
	_bind_joint_limitation(p_index, p_joint);
	notify_property_list_changed();
	_update_joint_limitation(p_index, p_joint);
}

Ref<JointLimitation3D> IterateIK3D::get_joint_limitation(int p_index, int p_joint) const {
	ERR_FAIL_INDEX_V(p_index, (int)settings.size(), Ref<JointLimitation3D>());
	const LocalVector<IterateIK3DJointSetting *> &joint_settings = iterate_settings[p_index]->joint_settings;
	ERR_FAIL_INDEX_V(p_joint, (int)joint_settings.size(), Ref<JointLimitation3D>());
	return joint_settings[p_joint]->limitation;
}

void IterateIK3D::set_joint_limitation_right_axis(int p_index, int p_joint, SecondaryDirection p_direction) {
	ERR_FAIL_INDEX(p_index, (int)settings.size());
	const LocalVector<IterateIK3DJointSetting *> &joint_settings = iterate_settings[p_index]->joint_settings;
	ERR_FAIL_INDEX(p_joint, (int)joint_settings.size());
	joint_settings[p_joint]->limitation_right_axis = p_direction;
	notify_property_list_changed();
	_update_joint_limitation(p_index, p_joint);
}

IKModifier3D::SecondaryDirection IterateIK3D::get_joint_limitation_right_axis(int p_index, int p_joint) const {
	ERR_FAIL_INDEX_V(p_index, (int)settings.size(), SECONDARY_DIRECTION_NONE);
	const LocalVector<IterateIK3DJointSetting *> &joint_settings = iterate_settings[p_index]->joint_settings;
	ERR_FAIL_INDEX_V(p_joint, (int)joint_settings.size(), SECONDARY_DIRECTION_NONE);
	return joint_settings[p_joint]->limitation_right_axis;
}

void IterateIK3D::set_joint_limitation_right_axis_vector(int p_index, int p_joint, const Vector3 &p_vector) {
	ERR_FAIL_INDEX(p_index, (int)settings.size());
	const LocalVector<IterateIK3DJointSetting *> &joint_settings = iterate_settings[p_index]->joint_settings;
	ERR_FAIL_INDEX(p_joint, (int)joint_settings.size());
	joint_settings[p_joint]->limitation_right_axis_vector = p_vector;
	_update_joint_limitation(p_index, p_joint);
}

Vector3 IterateIK3D::get_joint_limitation_right_axis_vector(int p_index, int p_joint) const {
	ERR_FAIL_INDEX_V(p_index, (int)settings.size(), Vector3());
	const LocalVector<IterateIK3DJointSetting *> &joint_settings = iterate_settings[p_index]->joint_settings;
	ERR_FAIL_INDEX_V(p_joint, (int)joint_settings.size(), Vector3());
	return joint_settings[p_joint]->get_limitation_right_axis_vector();
}

void IterateIK3D::set_joint_limitation_rotation_offset(int p_index, int p_joint, const Quaternion &p_offset) {
	ERR_FAIL_INDEX(p_index, (int)settings.size());
	const LocalVector<IterateIK3DJointSetting *> &joint_settings = iterate_settings[p_index]->joint_settings;
	ERR_FAIL_INDEX(p_joint, (int)joint_settings.size());
	joint_settings[p_joint]->limitation_rotation_offset = p_offset;
	_update_joint_limitation(p_index, p_joint);
}

Quaternion IterateIK3D::get_joint_limitation_rotation_offset(int p_index, int p_joint) const {
	ERR_FAIL_INDEX_V(p_index, (int)settings.size(), Quaternion());
	const LocalVector<IterateIK3DJointSetting *> &joint_settings = iterate_settings[p_index]->joint_settings;
	ERR_FAIL_INDEX_V(p_joint, (int)joint_settings.size(), Quaternion());
	return joint_settings[p_joint]->limitation_rotation_offset;
}

void IterateIK3D::_set_joint_count(int p_index, int p_count) {
	_unbind_joint_limitations(p_index);
	LocalVector<IterateIK3DJointSetting *> &joint_settings = iterate_settings[p_index]->joint_settings;
	int delta = p_count - joint_settings.size();
	if (delta < 0) {
		for (int i = delta; i < 0; i++) {
			memdelete(joint_settings[joint_settings.size() + i]);
			joint_settings[joint_settings.size() + i] = nullptr;
		}
	}
	joint_settings.resize(p_count);
	delta++;
	if (delta > 1) {
		for (int i = 1; i < delta; i++) {
			joint_settings[p_count - i] = memnew(IterateIK3DJointSetting);
		}
	}
}

void IterateIK3D::_validate_axis(Skeleton3D *p_skeleton, int p_index, int p_joint) const {
	RotationAxis axis = iterate_settings[p_index]->joint_settings[p_joint]->rotation_axis;
	if (axis == ROTATION_AXIS_ALL) {
		return;
	}
	Vector3 rot = get_joint_rotation_axis_vector(p_index, p_joint).normalized();
	Vector3 fwd;
	if (p_joint < (int)iterate_settings[p_index]->joints.size() - 1) {
		fwd = p_skeleton->get_bone_rest(iterate_settings[p_index]->joints[p_joint + 1].bone).origin;
	} else if (iterate_settings[p_index]->extend_end_bone) {
		fwd = IKModifier3D::get_bone_axis(p_skeleton, iterate_settings[p_index]->end_bone.bone, iterate_settings[p_index]->end_bone_direction, mutable_bone_axes);
		if (fwd.is_zero_approx()) {
			return;
		}
	}
	fwd.normalize();
	if (Math::is_equal_approx(Math::abs(rot.dot(fwd)), 1)) {
		WARN_PRINT_ED("Setting: " + itos(p_index) + " Joint: " + itos(p_joint) + ": Rotation axis and forward vector are colinear. This is not advised as it may cause unwanted rotation.");
	}
}

void IterateIK3D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_max_iterations", "max_iterations"), &IterateIK3D::set_max_iterations);
	ClassDB::bind_method(D_METHOD("get_max_iterations"), &IterateIK3D::get_max_iterations);
	ClassDB::bind_method(D_METHOD("set_min_distance", "min_distance"), &IterateIK3D::set_min_distance);
	ClassDB::bind_method(D_METHOD("get_min_distance"), &IterateIK3D::get_min_distance);
	ClassDB::bind_method(D_METHOD("set_angular_delta_limit", "angular_delta_limit"), &IterateIK3D::set_angular_delta_limit);
	ClassDB::bind_method(D_METHOD("get_angular_delta_limit"), &IterateIK3D::get_angular_delta_limit);
	ClassDB::bind_method(D_METHOD("set_deterministic", "deterministic"), &IterateIK3D::set_deterministic);
	ClassDB::bind_method(D_METHOD("is_deterministic"), &IterateIK3D::is_deterministic);

	// Setting.
	ClassDB::bind_method(D_METHOD("set_target_node", "index", "target_node"), &IterateIK3D::set_target_node);
	ClassDB::bind_method(D_METHOD("get_target_node", "index"), &IterateIK3D::get_target_node);

	// Individual joints.
	ClassDB::bind_method(D_METHOD("set_joint_rotation_axis", "index", "joint", "axis"), &IterateIK3D::set_joint_rotation_axis);
	ClassDB::bind_method(D_METHOD("get_joint_rotation_axis", "index", "joint"), &IterateIK3D::get_joint_rotation_axis);
	ClassDB::bind_method(D_METHOD("set_joint_rotation_axis_vector", "index", "joint", "axis_vector"), &IterateIK3D::set_joint_rotation_axis_vector);
	ClassDB::bind_method(D_METHOD("get_joint_rotation_axis_vector", "index", "joint"), &IterateIK3D::get_joint_rotation_axis_vector);
	ClassDB::bind_method(D_METHOD("set_joint_limitation", "index", "joint", "limitation"), &IterateIK3D::set_joint_limitation);
	ClassDB::bind_method(D_METHOD("get_joint_limitation", "index", "joint"), &IterateIK3D::get_joint_limitation);
	ClassDB::bind_method(D_METHOD("set_joint_limitation_right_axis", "index", "joint", "direction"), &IterateIK3D::set_joint_limitation_right_axis);
	ClassDB::bind_method(D_METHOD("get_joint_limitation_right_axis", "index", "joint"), &IterateIK3D::get_joint_limitation_right_axis);
	ClassDB::bind_method(D_METHOD("set_joint_limitation_right_axis_vector", "index", "joint", "vector"), &IterateIK3D::set_joint_limitation_right_axis_vector);
	ClassDB::bind_method(D_METHOD("get_joint_limitation_right_axis_vector", "index", "joint"), &IterateIK3D::get_joint_limitation_right_axis_vector);
	ClassDB::bind_method(D_METHOD("set_joint_limitation_rotation_offset", "index", "joint", "offset"), &IterateIK3D::set_joint_limitation_rotation_offset);
	ClassDB::bind_method(D_METHOD("get_joint_limitation_rotation_offset", "index", "joint"), &IterateIK3D::get_joint_limitation_rotation_offset);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "max_iterations", PROPERTY_HINT_RANGE, "0,100,or_greater"), "set_max_iterations", "get_max_iterations");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "min_distance", PROPERTY_HINT_RANGE, "0,1,0.001,or_greater"), "set_min_distance", "get_min_distance");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "angular_delta_limit", PROPERTY_HINT_RANGE, "0,180,0.001,radians_as_degrees"), "set_angular_delta_limit", "get_angular_delta_limit");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "deterministic"), "set_deterministic", "is_deterministic");
	ADD_ARRAY_COUNT("Settings", "setting_count", "set_setting_count", "get_setting_count", "settings/");
}

void IterateIK3D::_clear_joints(int p_index) {
	IterateIK3DSetting *setting = iterate_settings[p_index];
	if (!setting) {
		return;
	}
	_unbind_joint_limitations(p_index);
	for (uint32_t i = 0; i < setting->solver_info_list.size(); i++) {
		if (setting->solver_info_list[i]) {
			memdelete(setting->solver_info_list[i]);
			setting->solver_info_list[i] = nullptr;
		}
	}
	setting->solver_info_list.clear();
	setting->solver_info_list.resize_initialized(setting->joints.size());
	_bind_joint_limitations(p_index);
}

void IterateIK3D::_init_joints(Skeleton3D *p_skeleton, int p_index) {
	IterateIK3DSetting *setting = iterate_settings[p_index];
	if (!setting) {
		return;
	}
	cached_space = p_skeleton->get_global_transform_interpolated();
	if (setting->simulation_dirty) {
		_clear_joints(p_index);
		setting->init_joints(p_skeleton, mutable_bone_axes);
		setting->simulation_dirty = false;
	} else if (deterministic) {
		setting->init_joints(p_skeleton, mutable_bone_axes);
	}

	if (mutable_bone_axes) {
#ifdef TOOLS_ENABLED
		_update_mutable_info();
#endif // TOOLS_ENABLED
		_update_bone_axis(p_skeleton, p_index);
	}
	setting->simulated = false;
}

void IterateIK3D::_make_simulation_dirty(int p_index) {
	IterateIK3DSetting *setting = iterate_settings[p_index];
	if (!setting) {
		return;
	}
	setting->simulation_dirty = true;
#ifdef TOOLS_ENABLED
	if (!mutable_bone_axes) {
		_make_gizmo_dirty();
	}
#endif // TOOLS_ENABLED
}

void IterateIK3D::_update_bone_axis(Skeleton3D *p_skeleton, int p_index) {
#ifdef TOOLS_ENABLED
	bool changed = false;
#endif // TOOLS_ENABLED
	IterateIK3DSetting *setting = iterate_settings[p_index];
	const LocalVector<BoneJoint> &joints = setting->joints;
	const LocalVector<IKModifier3DSolverInfo *> &solver_info_list = setting->solver_info_list;
	int len = (int)solver_info_list.size() - 1;
	for (int j = 0; j < len; j++) {
		IterateIK3DJointSetting *joint_setting = setting->joint_settings[j];
		if (!joint_setting || !solver_info_list[j]) {
			continue;
		}
		Vector3 axis = p_skeleton->get_bone_pose(joints[j + 1].bone).origin;
		if (axis.is_zero_approx()) {
			continue;
		}
		// Less computing.
#ifdef TOOLS_ENABLED
		if (!changed) {
			Vector3 old_v = solver_info_list[j]->forward_vector;
			solver_info_list[j]->forward_vector = snap_vector_to_plane(joint_setting->get_rotation_axis_vector(), axis.normalized());
			changed = changed || !old_v.is_equal_approx(solver_info_list[j]->forward_vector);
			float old_l = solver_info_list[j]->length;
			solver_info_list[j]->length = axis.length();
			changed = changed || !Math::is_equal_approx(old_l, solver_info_list[j]->length);
		} else {
			solver_info_list[j]->forward_vector = snap_vector_to_plane(joint_setting->get_rotation_axis_vector(), axis.normalized());
			solver_info_list[j]->length = axis.length();
		}
#else
		solver_info_list[j]->forward_vector = snap_vector_to_plane(joint_setting->get_rotation_axis_vector(), axis.normalized());
		solver_info_list[j]->length = axis.length();
#endif // TOOLS_ENABLED
	}
	if (setting->extend_end_bone && len >= 0) {
		IterateIK3DJointSetting *joint_setting = setting->joint_settings[len];
		if (joint_setting && solver_info_list[len]) {
			Vector3 axis = IKModifier3D::get_bone_axis(p_skeleton, setting->end_bone.bone, setting->end_bone_direction, mutable_bone_axes);
			if (!axis.is_zero_approx()) {
				solver_info_list[len]->forward_vector = snap_vector_to_plane(joint_setting->get_rotation_axis_vector(), axis.normalized());
				solver_info_list[len]->length = setting->end_bone_length;
			}
		}
	}
#ifdef TOOLS_ENABLED
	if (changed) {
		_make_gizmo_dirty();
	}
#endif // TOOLS_ENABLED
}

void IterateIK3D::_process_ik(Skeleton3D *p_skeleton, double p_delta) {
	min_distance_squared = min_distance * min_distance;
	for (uint32_t i = 0; i < settings.size(); i++) {
		_init_joints(p_skeleton, i);
		Node3D *target = Object::cast_to<Node3D>(get_node_or_null(iterate_settings[i]->target_node));
		if (!target || iterate_settings[i]->chain.is_empty()) {
			continue; // Abort.
		}
		iterate_settings[i]->cache_current_joint_rotations(p_skeleton); // Iterate over first to detect parent (outside of the chain) bone pose changes.

		Vector3 destination = cached_space.affine_inverse().xform(target->get_global_transform_interpolated().origin);
		_process_joints(p_delta, p_skeleton, iterate_settings[i], destination);
	}
}

void IterateIK3D::_process_joints(double p_delta, Skeleton3D *p_skeleton, IterateIK3DSetting *p_setting, const Vector3 &p_destination) {
	double distance_to_target_sq = INFINITY;
	int iteration_count = 0;

	// To prevent oscillation, if it has been processed at least once and target was reached, abort iterating.
	if (p_setting->simulated) {
		distance_to_target_sq = p_setting->chain[p_setting->chain.size() - 1].distance_squared_to(p_destination);
	}

	while (distance_to_target_sq > min_distance_squared && iteration_count < max_iterations) {
		// Solve the IK for this iteration.
		_solve_iteration(p_delta, p_skeleton, p_setting, p_destination);

		// Update virtual bone rest/poses.
		p_setting->cache_current_joint_rotations(p_skeleton, angular_delta_limit);
		distance_to_target_sq = p_setting->chain[p_setting->chain.size() - 1].distance_squared_to(p_destination);
		iteration_count++;
	}

	// Apply the virtual bone rest/poses to the actual bones.
	for (uint32_t i = 0; i < p_setting->solver_info_list.size(); i++) {
		IKModifier3DSolverInfo *solver_info = p_setting->solver_info_list[i];
		if (!solver_info || Math::is_zero_approx(solver_info->length)) {
			continue;
		}
		p_skeleton->set_bone_pose_rotation(p_setting->joints[i].bone, solver_info->current_lpose);
	}

	p_setting->simulated = true;
}

void IterateIK3D::_solve_iteration(double p_delta, Skeleton3D *p_skeleton, IterateIK3DSetting *p_setting, const Vector3 &p_destination) {
	//
}

void IterateIK3D::_update_joint_limitation(int p_index, int p_joint) {
	ERR_FAIL_INDEX(p_index, (int)iterate_settings.size());
	iterate_settings[p_index]->simulated = false;
	const LocalVector<IterateIK3DJointSetting *> &joint_settings = iterate_settings[p_index]->joint_settings;
	ERR_FAIL_INDEX(p_joint, (int)joint_settings.size()); // p_joint is unused directly, but need to identify bound index.
#ifdef TOOLS_ENABLED
	update_gizmos();
#endif // TOOLS_ENABLED
}

void IterateIK3D::_bind_joint_limitation(int p_index, int p_joint) {
	ERR_FAIL_INDEX(p_index, (int)iterate_settings.size());
	const LocalVector<IterateIK3DJointSetting *> &joint_settings = iterate_settings[p_index]->joint_settings;
	ERR_FAIL_INDEX(p_joint, (int)joint_settings.size());
	if (joint_settings[p_joint]->limitation.is_valid()) {
		joint_settings[p_joint]->limitation->connect_changed(callable_mp(this, &IterateIK3D::_update_joint_limitation).bind(p_index, p_joint));
	}
}

void IterateIK3D::_unbind_joint_limitation(int p_index, int p_joint) {
	ERR_FAIL_INDEX(p_index, (int)iterate_settings.size());
	const LocalVector<IterateIK3DJointSetting *> &joint_settings = iterate_settings[p_index]->joint_settings;
	ERR_FAIL_INDEX(p_joint, (int)joint_settings.size());
	if (joint_settings[p_joint]->limitation.is_valid()) {
		joint_settings[p_joint]->limitation->disconnect_changed(callable_mp(this, &IterateIK3D::_update_joint_limitation).bind(p_index, p_joint));
	}
}

void IterateIK3D::_bind_joint_limitations(int p_index) {
	for (uint32_t i = 0; i < iterate_settings[p_index]->joints.size(); i++) {
		if (iterate_settings[p_index]->joint_settings[i]->limitation.is_valid()) {
			iterate_settings[p_index]->joint_settings[i]->limitation->connect_changed(callable_mp(this, &IterateIK3D::_update_joint_limitation).bind(p_index, i));
		}
	}
}

void IterateIK3D::_unbind_joint_limitations(int p_index) {
	for (uint32_t i = 0; i < iterate_settings[p_index]->joint_settings.size(); i++) {
		if (iterate_settings[p_index]->joint_settings[i]->limitation.is_valid()) {
			iterate_settings[p_index]->joint_settings[i]->limitation->disconnect_changed(callable_mp(this, &IterateIK3D::_update_joint_limitation).bind(p_index, i));
		}
	}
}

#ifdef TOOLS_ENABLED
Vector3 IterateIK3D::get_bone_vector(int p_index, int p_joint) const {
	Skeleton3D *skeleton = get_skeleton();
	if (!skeleton) {
		return Vector3();
	}
	ERR_FAIL_INDEX_V(p_index, (int)settings.size(), Vector3());
	IterateIK3DSetting *setting = iterate_settings[p_index];
	if (!setting) {
		return Vector3();
	}
	const LocalVector<BoneJoint> &joints = setting->joints;
	ERR_FAIL_INDEX_V(p_joint, (int)joints.size(), Vector3());
	const LocalVector<IKModifier3DSolverInfo *> &solver_info_list = setting->solver_info_list;
	if (p_joint >= (int)solver_info_list.size() || !solver_info_list[p_joint]) {
		if (p_joint == (int)joints.size() - 1) {
			return IKModifier3D::get_bone_axis(skeleton, setting->end_bone.bone, setting->end_bone_direction, mutable_bone_axes) * setting->end_bone_length;
		}
		return mutable_bone_axes ? skeleton->get_bone_pose(joints[p_joint + 1].bone).origin : skeleton->get_bone_rest(joints[p_joint + 1].bone).origin;
	}
	return solver_info_list[p_joint]->forward_vector * solver_info_list[p_joint]->length;
}
#endif // TOOLS_ENABLED

IterateIK3D::~IterateIK3D() {
	for (uint32_t i = 0; i < iterate_settings.size(); i++) {
		_unbind_joint_limitations(i);
	}
	clear_settings();
}
