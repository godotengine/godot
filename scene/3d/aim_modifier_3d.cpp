/**************************************************************************/
/*  aim_modifier_3d.cpp                                                   */
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

#include "aim_modifier_3d.h"

bool AimModifier3D::_set(const StringName &p_path, const Variant &p_value) {
	String path = p_path;

	if (path.begins_with("settings/")) {
		int which = path.get_slicec('/', 1).to_int();
		String what = path.get_slicec('/', 2);
		ERR_FAIL_INDEX_V(which, settings.size(), false);

		if (what == "forward_axis") {
			set_forward_axis(which, static_cast<BoneAxis>((int)p_value));
		} else if (what == "use_euler") {
			set_use_euler(which, p_value);
		} else if (what == "primary_rotation_axis") {
			set_primary_rotation_axis(which, static_cast<Vector3::Axis>((int)p_value));
		} else if (what == "use_secondary_rotation") {
			set_use_secondary_rotation(which, p_value);
		} else {
			return false;
		}
	}
	return true;
}

bool AimModifier3D::_get(const StringName &p_path, Variant &r_ret) const {
	String path = p_path;

	if (path.begins_with("settings/")) {
		int which = path.get_slicec('/', 1).to_int();
		String what = path.get_slicec('/', 2);
		ERR_FAIL_INDEX_V(which, settings.size(), false);

		if (what == "forward_axis") {
			r_ret = (int)get_forward_axis(which);
		} else if (what == "use_euler") {
			r_ret = is_using_euler(which);
		} else if (what == "primary_rotation_axis") {
			r_ret = (int)get_primary_rotation_axis(which);
		} else if (what == "use_secondary_rotation") {
			r_ret = is_using_secondary_rotation(which);
		} else {
			return false;
		}
	}
	return true;
}

void AimModifier3D::_validate_dynamic_prop(PropertyInfo &p_property) const {
	PackedStringArray split = p_property.name.split("/");
	if (split.size() == 3 && split[0] == "settings" && (split[2] == "primary_rotation_axis" || split[2] == "use_secondary_rotation")) {
		int which = split[1].to_int();
		if (!is_using_euler(which)) {
			p_property.usage = PROPERTY_USAGE_NONE;
		}
	}
}

void AimModifier3D::_get_property_list(List<PropertyInfo> *p_list) const {
	BoneConstraint3D::get_property_list(p_list);
	LocalVector<PropertyInfo> props;
	for (int i = 0; i < settings.size(); i++) {
		String path = "settings/" + itos(i) + "/";
		props.push_back(PropertyInfo(Variant::INT, path + "forward_axis", PROPERTY_HINT_ENUM, "+X,-X,+Y,-Y,+Z,-Z"));
		props.push_back(PropertyInfo(Variant::BOOL, path + "use_euler"));
		props.push_back(PropertyInfo(Variant::INT, path + "primary_rotation_axis", PROPERTY_HINT_ENUM, "X,Y,Z"));
		props.push_back(PropertyInfo(Variant::BOOL, path + "use_secondary_rotation"));
	}
	for (PropertyInfo &p : props) {
		_validate_dynamic_prop(p);
		p_list->push_back(p);
	}
}

PackedStringArray AimModifier3D::get_configuration_warnings() const {
	PackedStringArray warnings = BoneConstraint3D::get_configuration_warnings();
	for (int i = 0; i < settings.size(); i++) {
		if (is_using_euler(i) && get_axis_from_bone_axis(get_forward_axis(i)) == get_primary_rotation_axis(i)) {
			warnings.push_back(vformat(RTR("Forward axis and primary rotation axis must not be parallel in setting %s."), itos(i)));
		}
	}

	return warnings;
}

void AimModifier3D::_validate_setting(int p_index) {
	settings.write[p_index] = memnew(AimModifier3DSetting);
}

void AimModifier3D::set_forward_axis(int p_index, BoneAxis p_axis) {
	ERR_FAIL_INDEX(p_index, settings.size());
	AimModifier3DSetting *setting = static_cast<AimModifier3DSetting *>(settings[p_index]);
	setting->forward_axis = p_axis;
	update_configuration_warnings();
}

SkeletonModifier3D::BoneAxis AimModifier3D::get_forward_axis(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, settings.size(), BONE_AXIS_PLUS_Y);
	AimModifier3DSetting *setting = static_cast<AimModifier3DSetting *>(settings[p_index]);
	return setting->forward_axis;
}

void AimModifier3D::set_use_euler(int p_index, bool p_enabled) {
	ERR_FAIL_INDEX(p_index, settings.size());
	AimModifier3DSetting *setting = static_cast<AimModifier3DSetting *>(settings[p_index]);
	setting->use_euler = p_enabled;
	notify_property_list_changed();
	update_configuration_warnings();
}

bool AimModifier3D::is_using_euler(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, settings.size(), false);
	AimModifier3DSetting *setting = static_cast<AimModifier3DSetting *>(settings[p_index]);
	return setting->use_euler;
}

void AimModifier3D::set_primary_rotation_axis(int p_index, Vector3::Axis p_axis) {
	ERR_FAIL_INDEX(p_index, settings.size());
	AimModifier3DSetting *setting = static_cast<AimModifier3DSetting *>(settings[p_index]);
	setting->primary_rotation_axis = p_axis;
	update_configuration_warnings();
}

Vector3::Axis AimModifier3D::get_primary_rotation_axis(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, settings.size(), Vector3::AXIS_X);
	AimModifier3DSetting *setting = static_cast<AimModifier3DSetting *>(settings[p_index]);
	return setting->primary_rotation_axis;
}

void AimModifier3D::set_use_secondary_rotation(int p_index, bool p_enabled) {
	ERR_FAIL_INDEX(p_index, settings.size());
	AimModifier3DSetting *setting = static_cast<AimModifier3DSetting *>(settings[p_index]);
	setting->use_secondary_rotation = p_enabled;
}

bool AimModifier3D::is_using_secondary_rotation(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, settings.size(), false);
	AimModifier3DSetting *setting = static_cast<AimModifier3DSetting *>(settings[p_index]);
	return setting->use_secondary_rotation;
}

void AimModifier3D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_forward_axis", "index", "axis"), &AimModifier3D::set_forward_axis);
	ClassDB::bind_method(D_METHOD("get_forward_axis", "index"), &AimModifier3D::get_forward_axis);
	ClassDB::bind_method(D_METHOD("set_use_euler", "index", "enabled"), &AimModifier3D::set_use_euler);
	ClassDB::bind_method(D_METHOD("is_using_euler", "index"), &AimModifier3D::is_using_euler);
	ClassDB::bind_method(D_METHOD("set_primary_rotation_axis", "index", "axis"), &AimModifier3D::set_primary_rotation_axis);
	ClassDB::bind_method(D_METHOD("get_primary_rotation_axis", "index"), &AimModifier3D::get_primary_rotation_axis);
	ClassDB::bind_method(D_METHOD("set_use_secondary_rotation", "index", "enabled"), &AimModifier3D::set_use_secondary_rotation);
	ClassDB::bind_method(D_METHOD("is_using_secondary_rotation", "index"), &AimModifier3D::is_using_secondary_rotation);

	ADD_ARRAY_COUNT("Settings", "setting_count", "set_setting_count", "get_setting_count", "settings/");
}

void AimModifier3D::_process_constraint(int p_index, Skeleton3D *p_skeleton, int p_apply_bone, int p_reference_bone, float p_amount) {
	if (p_apply_bone == p_reference_bone) {
		ERR_PRINT_ONCE_ED(vformat("In setting %s, the reference bone must not be same with the apply bone.", itos(p_index)));
		return;
	}

	AimModifier3DSetting *setting = static_cast<AimModifier3DSetting *>(settings[p_index]);

	// Prepare forward_vector and rest.
	Vector3 reference_origin = p_skeleton->get_bone_global_pose(p_reference_bone).origin;
	Transform3D src_bone_rest = p_skeleton->get_bone_rest(p_apply_bone);
	Transform3D bone_rest_space;
	int parent_bone = p_skeleton->get_bone_parent(p_apply_bone);
	if (parent_bone < 0) {
		bone_rest_space.translate_local(src_bone_rest.origin);
	} else {
		bone_rest_space = p_skeleton->get_bone_global_pose(parent_bone);
		bone_rest_space.translate_local(src_bone_rest.origin);
	}
	Vector3 forward_vector = bone_rest_space.basis.get_rotation_quaternion().xform_inv(reference_origin - bone_rest_space.origin);
	if (forward_vector.is_zero_approx()) {
		return;
	}
	forward_vector.normalize();

	// Calculate look at rotation.
	Quaternion destination;
	if (setting->use_euler) {
		Vector3 current_vector = LookAtModifier3D::get_basis_vector_from_bone_axis(src_bone_rest.basis, setting->forward_axis).normalized();
		Vector2 src_vec2 = LookAtModifier3D::get_projection_vector(src_bone_rest.basis.xform_inv(forward_vector), setting->primary_rotation_axis).normalized();
		Vector2 dst_vec2 = LookAtModifier3D::get_projection_vector(src_bone_rest.basis.xform_inv(current_vector), setting->primary_rotation_axis).normalized();
		real_t calculated_angle = src_vec2.angle_to(dst_vec2);
		Transform3D primary_result = src_bone_rest.rotated_local(get_vector_from_axis(setting->primary_rotation_axis), calculated_angle);
		Transform3D current_result = primary_result;
		if (setting->use_secondary_rotation) {
			Vector3::Axis secondary_rotation_axis = LookAtModifier3D::get_secondary_rotation_axis(setting->forward_axis, setting->primary_rotation_axis);
			current_vector = LookAtModifier3D::get_basis_vector_from_bone_axis(primary_result.basis, setting->forward_axis).normalized();
			src_vec2 = LookAtModifier3D::get_projection_vector(primary_result.basis.xform_inv(forward_vector), secondary_rotation_axis).normalized();
			dst_vec2 = LookAtModifier3D::get_projection_vector(primary_result.basis.xform_inv(current_vector), secondary_rotation_axis).normalized();
			calculated_angle = src_vec2.angle_to(dst_vec2);
			current_result = primary_result.rotated_local(get_vector_from_axis(secondary_rotation_axis), calculated_angle);
		}
		destination = current_result.basis.get_rotation_quaternion();
	} else {
		Vector3 current_vector = LookAtModifier3D::get_basis_vector_from_bone_axis(src_bone_rest.basis, setting->forward_axis).normalized();
		destination = Quaternion(current_vector, forward_vector) * src_bone_rest.basis.get_rotation_quaternion();
	}

	p_skeleton->set_bone_pose_rotation(p_apply_bone, p_skeleton->get_bone_pose_rotation(p_apply_bone).slerp(destination, p_amount));
}

AimModifier3D::~AimModifier3D() {
	clear_settings();
}
