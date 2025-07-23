/**************************************************************************/
/*  convert_transform_modifier_3d.cpp                                     */
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

#include "convert_transform_modifier_3d.h"

constexpr const char *HINT_POSITION = "-10,10,0.01,or_greater,or_less,suffix:m";
constexpr const char *HINT_ROTATION = "-180,180,0.01,radians_as_degrees";
constexpr const char *HINT_SCALE = "0,10,0.01,or_greater";

bool ConvertTransformModifier3D::_set(const StringName &p_path, const Variant &p_value) {
	String path = p_path;

	if (path.begins_with("settings/")) {
		int which = path.get_slicec('/', 1).to_int();
		String where = path.get_slicec('/', 2);
		ERR_FAIL_INDEX_V(which, settings.size(), false);
		String what = path.get_slicec('/', 3);

		if (where == "apply") {
			if (what == "transform_mode") {
				set_apply_transform_mode(which, static_cast<TransformMode>((int)p_value));
			} else if (what == "axis") {
				set_apply_axis(which, static_cast<Vector3::Axis>((int)p_value));
			} else if (what == "range_min") {
				set_apply_range_min(which, p_value);
			} else if (what == "range_max") {
				set_apply_range_max(which, p_value);
			} else {
				return false;
			}
		} else if (where == "reference") {
			if (what == "transform_mode") {
				set_reference_transform_mode(which, static_cast<TransformMode>((int)p_value));
			} else if (what == "axis") {
				set_reference_axis(which, static_cast<Vector3::Axis>((int)p_value));
			} else if (what == "range_min") {
				set_reference_range_min(which, p_value);
			} else if (what == "range_max") {
				set_reference_range_max(which, p_value);
			} else {
				return false;
			}
		} else if (where == "relative") {
			set_relative(which, p_value);
		} else if (where == "additive") {
			set_additive(which, p_value);
		} else {
			return false;
		}
	}
	return true;
}

bool ConvertTransformModifier3D::_get(const StringName &p_path, Variant &r_ret) const {
	String path = p_path;

	if (path.begins_with("settings/")) {
		int which = path.get_slicec('/', 1).to_int();
		String where = path.get_slicec('/', 2);
		ERR_FAIL_INDEX_V(which, settings.size(), false);
		String what = path.get_slicec('/', 3);

		if (where == "apply") {
			if (what == "transform_mode") {
				r_ret = (int)get_apply_transform_mode(which);
			} else if (what == "axis") {
				r_ret = (int)get_apply_axis(which);
			} else if (what == "range_min") {
				r_ret = get_apply_range_min(which);
			} else if (what == "range_max") {
				r_ret = get_apply_range_max(which);
			} else {
				return false;
			}
		} else if (where == "reference") {
			if (what == "transform_mode") {
				r_ret = (int)get_reference_transform_mode(which);
			} else if (what == "axis") {
				r_ret = (int)get_reference_axis(which);
			} else if (what == "range_min") {
				r_ret = get_reference_range_min(which);
			} else if (what == "range_max") {
				r_ret = get_reference_range_max(which);
			} else {
				return false;
			}
		} else if (where == "relative") {
			r_ret = is_relative(which);
		} else if (where == "additive") {
			r_ret = is_additive(which);
		} else {
			return false;
		}
	}
	return true;
}

void ConvertTransformModifier3D::_get_property_list(List<PropertyInfo> *p_list) const {
	BoneConstraint3D::get_property_list(p_list);

	for (int i = 0; i < settings.size(); i++) {
		String path = "settings/" + itos(i) + "/";

		String hint_apply_range;
		if (get_apply_transform_mode(i) == TRANSFORM_MODE_POSITION) {
			hint_apply_range = HINT_POSITION;
		} else if (get_apply_transform_mode(i) == TRANSFORM_MODE_ROTATION) {
			hint_apply_range = HINT_ROTATION;
		} else {
			hint_apply_range = HINT_SCALE;
		}
		p_list->push_back(PropertyInfo(Variant::INT, path + "apply/transform_mode", PROPERTY_HINT_ENUM, "Position,Rotation,Scale"));
		p_list->push_back(PropertyInfo(Variant::INT, path + "apply/axis", PROPERTY_HINT_ENUM, "X,Y,Z"));
		p_list->push_back(PropertyInfo(Variant::FLOAT, path + "apply/range_min", PROPERTY_HINT_RANGE, hint_apply_range));
		p_list->push_back(PropertyInfo(Variant::FLOAT, path + "apply/range_max", PROPERTY_HINT_RANGE, hint_apply_range));

		String hint_reference_range;
		if (get_reference_transform_mode(i) == TRANSFORM_MODE_POSITION) {
			hint_reference_range = HINT_POSITION;
		} else if (get_reference_transform_mode(i) == TRANSFORM_MODE_ROTATION) {
			hint_reference_range = HINT_ROTATION;
		} else {
			hint_reference_range = HINT_SCALE;
		}
		p_list->push_back(PropertyInfo(Variant::INT, path + "reference/transform_mode", PROPERTY_HINT_ENUM, "Position,Rotation,Scale"));
		p_list->push_back(PropertyInfo(Variant::INT, path + "reference/axis", PROPERTY_HINT_ENUM, "X,Y,Z"));
		p_list->push_back(PropertyInfo(Variant::FLOAT, path + "reference/range_min", PROPERTY_HINT_RANGE, hint_reference_range));
		p_list->push_back(PropertyInfo(Variant::FLOAT, path + "reference/range_max", PROPERTY_HINT_RANGE, hint_reference_range));

		p_list->push_back(PropertyInfo(Variant::BOOL, path + "relative"));
		p_list->push_back(PropertyInfo(Variant::BOOL, path + "additive"));
	}
}

void ConvertTransformModifier3D::_validate_setting(int p_index) {
	settings.write[p_index] = memnew(ConvertTransform3DSetting);
}

void ConvertTransformModifier3D::set_apply_transform_mode(int p_index, TransformMode p_transform_mode) {
	ERR_FAIL_INDEX(p_index, settings.size());
	ConvertTransform3DSetting *setting = static_cast<ConvertTransform3DSetting *>(settings[p_index]);
	setting->apply_transform_mode = p_transform_mode;
	notify_property_list_changed();
}

ConvertTransformModifier3D::TransformMode ConvertTransformModifier3D::get_apply_transform_mode(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, settings.size(), TRANSFORM_MODE_POSITION);
	ConvertTransform3DSetting *setting = static_cast<ConvertTransform3DSetting *>(settings[p_index]);
	return setting->apply_transform_mode;
}

void ConvertTransformModifier3D::set_apply_axis(int p_index, Vector3::Axis p_axis) {
	ERR_FAIL_INDEX(p_index, settings.size());
	ConvertTransform3DSetting *setting = static_cast<ConvertTransform3DSetting *>(settings[p_index]);
	setting->apply_axis = p_axis;
}

Vector3::Axis ConvertTransformModifier3D::get_apply_axis(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, settings.size(), Vector3::AXIS_X);
	ConvertTransform3DSetting *setting = static_cast<ConvertTransform3DSetting *>(settings[p_index]);
	return setting->apply_axis;
}

void ConvertTransformModifier3D::set_apply_range_min(int p_index, float p_range_min) {
	ERR_FAIL_INDEX(p_index, settings.size());
	ConvertTransform3DSetting *setting = static_cast<ConvertTransform3DSetting *>(settings[p_index]);
	setting->apply_range_min = p_range_min;
}

float ConvertTransformModifier3D::get_apply_range_min(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, settings.size(), 0);
	ConvertTransform3DSetting *setting = static_cast<ConvertTransform3DSetting *>(settings[p_index]);
	return setting->apply_range_min;
}

void ConvertTransformModifier3D::set_apply_range_max(int p_index, float p_range_max) {
	ERR_FAIL_INDEX(p_index, settings.size());
	ConvertTransform3DSetting *setting = static_cast<ConvertTransform3DSetting *>(settings[p_index]);
	setting->apply_range_max = p_range_max;
}

float ConvertTransformModifier3D::get_apply_range_max(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, settings.size(), 0);
	ConvertTransform3DSetting *setting = static_cast<ConvertTransform3DSetting *>(settings[p_index]);
	return setting->apply_range_max;
}

void ConvertTransformModifier3D::set_reference_transform_mode(int p_index, TransformMode p_transform_mode) {
	ERR_FAIL_INDEX(p_index, settings.size());
	ConvertTransform3DSetting *setting = static_cast<ConvertTransform3DSetting *>(settings[p_index]);
	setting->reference_transform_mode = p_transform_mode;
	notify_property_list_changed();
}

ConvertTransformModifier3D::TransformMode ConvertTransformModifier3D::get_reference_transform_mode(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, settings.size(), TRANSFORM_MODE_POSITION);
	ConvertTransform3DSetting *setting = static_cast<ConvertTransform3DSetting *>(settings[p_index]);
	return setting->reference_transform_mode;
}

void ConvertTransformModifier3D::set_reference_axis(int p_index, Vector3::Axis p_axis) {
	ERR_FAIL_INDEX(p_index, settings.size());
	ConvertTransform3DSetting *setting = static_cast<ConvertTransform3DSetting *>(settings[p_index]);
	setting->reference_axis = p_axis;
}

Vector3::Axis ConvertTransformModifier3D::get_reference_axis(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, settings.size(), Vector3::AXIS_X);
	ConvertTransform3DSetting *setting = static_cast<ConvertTransform3DSetting *>(settings[p_index]);
	return setting->reference_axis;
}

void ConvertTransformModifier3D::set_reference_range_min(int p_index, float p_range_min) {
	ERR_FAIL_INDEX(p_index, settings.size());
	ConvertTransform3DSetting *setting = static_cast<ConvertTransform3DSetting *>(settings[p_index]);
	setting->reference_range_min = p_range_min;
}

float ConvertTransformModifier3D::get_reference_range_min(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, settings.size(), 0);
	ConvertTransform3DSetting *setting = static_cast<ConvertTransform3DSetting *>(settings[p_index]);
	return setting->reference_range_min;
}

void ConvertTransformModifier3D::set_reference_range_max(int p_index, float p_range_max) {
	ERR_FAIL_INDEX(p_index, settings.size());
	ConvertTransform3DSetting *setting = static_cast<ConvertTransform3DSetting *>(settings[p_index]);
	setting->reference_range_max = p_range_max;
}

float ConvertTransformModifier3D::get_reference_range_max(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, settings.size(), 0);
	ConvertTransform3DSetting *setting = static_cast<ConvertTransform3DSetting *>(settings[p_index]);
	return setting->reference_range_max;
}

void ConvertTransformModifier3D::set_relative(int p_index, bool p_enabled) {
	ERR_FAIL_INDEX(p_index, settings.size());
	ConvertTransform3DSetting *setting = static_cast<ConvertTransform3DSetting *>(settings[p_index]);
	setting->relative = p_enabled;
}

bool ConvertTransformModifier3D::is_relative(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, settings.size(), 0);
	ConvertTransform3DSetting *setting = static_cast<ConvertTransform3DSetting *>(settings[p_index]);
	return setting->relative;
}

void ConvertTransformModifier3D::set_additive(int p_index, bool p_enabled) {
	ERR_FAIL_INDEX(p_index, settings.size());
	ConvertTransform3DSetting *setting = static_cast<ConvertTransform3DSetting *>(settings[p_index]);
	setting->additive = p_enabled;
}

bool ConvertTransformModifier3D::is_additive(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, settings.size(), 0);
	ConvertTransform3DSetting *setting = static_cast<ConvertTransform3DSetting *>(settings[p_index]);
	return setting->additive;
}

void ConvertTransformModifier3D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_apply_transform_mode", "index", "transform_mode"), &ConvertTransformModifier3D::set_apply_transform_mode);
	ClassDB::bind_method(D_METHOD("get_apply_transform_mode", "index"), &ConvertTransformModifier3D::get_apply_transform_mode);
	ClassDB::bind_method(D_METHOD("set_apply_axis", "index", "axis"), &ConvertTransformModifier3D::set_apply_axis);
	ClassDB::bind_method(D_METHOD("get_apply_axis", "index"), &ConvertTransformModifier3D::get_apply_axis);
	ClassDB::bind_method(D_METHOD("set_apply_range_min", "index", "range_min"), &ConvertTransformModifier3D::set_apply_range_min);
	ClassDB::bind_method(D_METHOD("get_apply_range_min", "index"), &ConvertTransformModifier3D::get_apply_range_min);
	ClassDB::bind_method(D_METHOD("set_apply_range_max", "index", "range_max"), &ConvertTransformModifier3D::set_apply_range_max);
	ClassDB::bind_method(D_METHOD("get_apply_range_max", "index"), &ConvertTransformModifier3D::get_apply_range_max);

	ClassDB::bind_method(D_METHOD("set_reference_transform_mode", "index", "transform_mode"), &ConvertTransformModifier3D::set_reference_transform_mode);
	ClassDB::bind_method(D_METHOD("get_reference_transform_mode", "index"), &ConvertTransformModifier3D::get_reference_transform_mode);
	ClassDB::bind_method(D_METHOD("set_reference_axis", "index", "axis"), &ConvertTransformModifier3D::set_reference_axis);
	ClassDB::bind_method(D_METHOD("get_reference_axis", "index"), &ConvertTransformModifier3D::get_reference_axis);
	ClassDB::bind_method(D_METHOD("set_reference_range_min", "index", "range_min"), &ConvertTransformModifier3D::set_reference_range_min);
	ClassDB::bind_method(D_METHOD("get_reference_range_min", "index"), &ConvertTransformModifier3D::get_reference_range_min);
	ClassDB::bind_method(D_METHOD("set_reference_range_max", "index", "range_max"), &ConvertTransformModifier3D::set_reference_range_max);
	ClassDB::bind_method(D_METHOD("get_reference_range_max", "index"), &ConvertTransformModifier3D::get_reference_range_max);

	ClassDB::bind_method(D_METHOD("set_relative", "index", "enabled"), &ConvertTransformModifier3D::set_relative);
	ClassDB::bind_method(D_METHOD("is_relative", "index"), &ConvertTransformModifier3D::is_relative);
	ClassDB::bind_method(D_METHOD("set_additive", "index", "enabled"), &ConvertTransformModifier3D::set_additive);
	ClassDB::bind_method(D_METHOD("is_additive", "index"), &ConvertTransformModifier3D::is_additive);

	ADD_ARRAY_COUNT("Settings", "setting_count", "set_setting_count", "get_setting_count", "settings/");

	BIND_ENUM_CONSTANT(TRANSFORM_MODE_POSITION);
	BIND_ENUM_CONSTANT(TRANSFORM_MODE_ROTATION);
	BIND_ENUM_CONSTANT(TRANSFORM_MODE_SCALE);
}

void ConvertTransformModifier3D::_process_constraint(int p_index, Skeleton3D *p_skeleton, int p_apply_bone, int p_reference_bone, float p_amount) {
	ConvertTransform3DSetting *setting = static_cast<ConvertTransform3DSetting *>(settings[p_index]);

	Transform3D destination = p_skeleton->get_bone_pose(p_reference_bone);
	if (setting->relative) {
		Vector3 scl_relative = destination.basis.get_scale() / p_skeleton->get_bone_rest(p_reference_bone).basis.get_scale();
		destination.basis = p_skeleton->get_bone_rest(p_reference_bone).basis.get_rotation_quaternion().inverse() * destination.basis.get_rotation_quaternion();
		destination.basis.scale_local(scl_relative);
		destination.origin = destination.origin - p_skeleton->get_bone_rest(p_reference_bone).origin;
	}

	// Retrieve point from reference.
	double point = 0.0;
	int axis = (int)setting->reference_axis;
	switch (setting->reference_transform_mode) {
		case TRANSFORM_MODE_POSITION: {
			point = destination.origin[axis];
		} break;
		case TRANSFORM_MODE_ROTATION: {
			Quaternion tgt_rot = destination.basis.get_rotation_quaternion();
			point = get_roll_angle(tgt_rot, get_vector_from_axis(setting->reference_axis));
		} break;
		case TRANSFORM_MODE_SCALE: {
			point = destination.basis.get_scale()[axis];
		} break;
	}
	// Convert point to apply.
	destination = p_skeleton->get_bone_pose(p_apply_bone);
	if (Math::is_equal_approx(setting->reference_range_min, setting->reference_range_max)) {
		point = point <= (double)setting->reference_range_min ? 0 : 1;
	} else {
		point = Math::inverse_lerp((double)setting->reference_range_min, (double)setting->reference_range_max, point);
	}
	point = Math::lerp((double)setting->apply_range_min, (double)setting->apply_range_max, CLAMP(point, 0, 1));
	axis = (int)setting->apply_axis;
	switch (setting->apply_transform_mode) {
		case TRANSFORM_MODE_POSITION: {
			if (setting->additive) {
				point = p_skeleton->get_bone_pose(p_apply_bone).origin[axis] + point;
			} else if (setting->relative) {
				point = p_skeleton->get_bone_rest(p_apply_bone).origin[axis] + point;
			}
			destination.origin[axis] = point;
		} break;
		case TRANSFORM_MODE_ROTATION: {
			Vector3 rot_axis = get_vector_from_axis(setting->apply_axis);
			Vector3 dest_scl = destination.basis.get_scale();
			if (influence < 1.0 || p_amount < 1.0) {
				point = CLAMP(point, CMP_EPSILON - Math::PI, Math::PI - CMP_EPSILON); // Hack to consistent slerp (interpolate_with) orientation since -180/180 deg rot is mixed in slerp.
			}
			Quaternion rot = Quaternion(rot_axis, point);
			if (setting->additive) {
				destination.basis = p_skeleton->get_bone_pose(p_apply_bone).basis.get_rotation_quaternion() * rot;
			} else if (setting->relative) {
				destination.basis = p_skeleton->get_bone_rest(p_apply_bone).basis.get_rotation_quaternion() * rot;
			} else {
				destination.basis = rot;
			}
			// Scale may not have meaning, but it might affect when it is negative.
			destination.basis.scale_local(dest_scl);
		} break;
		case TRANSFORM_MODE_SCALE: {
			Vector3 dest_scl = Vector3(1, 1, 1);
			if (setting->additive) {
				dest_scl = p_skeleton->get_bone_pose(p_apply_bone).basis.get_scale();
				dest_scl[axis] = dest_scl[axis] * point;
			} else if (setting->relative) {
				dest_scl = p_skeleton->get_bone_rest(p_apply_bone).basis.get_scale();
				dest_scl[axis] = dest_scl[axis] * point;
			} else {
				dest_scl = p_skeleton->get_bone_pose(p_apply_bone).basis.get_scale();
				dest_scl[axis] = point;
			}
			destination.basis = destination.basis.orthonormalized().scaled_local(dest_scl);
		} break;
	}
	// Process interpolation depends on the amount.
	destination = p_skeleton->get_bone_pose(p_apply_bone).interpolate_with(destination, p_amount);
	// Apply transform depends on the mode.
	switch (setting->apply_transform_mode) {
		case TRANSFORM_MODE_POSITION: {
			p_skeleton->set_bone_pose_position(p_apply_bone, destination.origin);
		} break;
		case TRANSFORM_MODE_ROTATION: {
			p_skeleton->set_bone_pose_rotation(p_apply_bone, destination.basis.get_rotation_quaternion());
		} break;
		case TRANSFORM_MODE_SCALE: {
			p_skeleton->set_bone_pose_scale(p_apply_bone, destination.basis.get_scale());
		} break;
	}
}

ConvertTransformModifier3D::~ConvertTransformModifier3D() {
	clear_settings();
}
