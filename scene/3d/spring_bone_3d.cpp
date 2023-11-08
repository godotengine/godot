/**************************************************************************/
/*  spring_bone_3d.cpp                                                    */
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

#include "spring_bone_3d.h"
#include "scene/scene_string_names.h"

#ifndef _3D_DISABLED

void SpringBone3D::_validate_property(PropertyInfo &p_property) const {
	if (p_property.name == "bone") {
		Skeleton3D *skeleton = get_skeleton();
		if (skeleton) {
			String names("--,");
			for (int i = 0; i < skeleton->get_bone_count(); i++) {
				if (i > 0) {
					names += ",";
				}
				names += skeleton->get_bone_name(i);
			}

			p_property.hint = PROPERTY_HINT_ENUM;
			p_property.hint_string = names;
		} else {
			p_property.hint = PROPERTY_HINT_NONE;
			p_property.hint_string = "";
		}
	} else if (p_property.name == "tail_bone") {
		Skeleton3D *skeleton = get_skeleton();
		if (skeleton && bone_id >= 0) {
			String names("--,");
			PackedInt32Array children_bones = skeleton->get_bone_children(bone_id);
			for (int i = 0; i < children_bones.size(); i++) {
				if (i > 0) {
					names += ",";
				}
				names += skeleton->get_bone_name(children_bones[i]);
			}

			p_property.hint = PROPERTY_HINT_ENUM;
			p_property.hint_string = names;
		} else {
			p_property.hint = PROPERTY_HINT_NONE;
			p_property.hint_string = "";
		}
	}
}

void SpringBone3D::_bind_methods() {
	GDVIRTUAL_BIND(_adjust_tail_position, "tail_position", "previous_tail_position", "bone_position", "tail_length");

	ClassDB::bind_method(D_METHOD("set_enabled", "enabled"), &SpringBone3D::set_enabled);
	ClassDB::bind_method(D_METHOD("is_enabled"), &SpringBone3D::is_enabled);

	ClassDB::bind_method(D_METHOD("set_skeleton_node", "skeleton_node"), &SpringBone3D::set_skeleton_node);
	ClassDB::bind_method(D_METHOD("get_skeleton_node"), &SpringBone3D::get_skeleton_node);

	ClassDB::bind_method(D_METHOD("set_bone", "bone"), &SpringBone3D::set_bone);
	ClassDB::bind_method(D_METHOD("get_bone"), &SpringBone3D::get_bone);

	ClassDB::bind_method(D_METHOD("set_stiffness", "stiffness"), &SpringBone3D::set_stiffness);
	ClassDB::bind_method(D_METHOD("get_stiffness"), &SpringBone3D::get_stiffness);

	ClassDB::bind_method(D_METHOD("set_damping", "damping"), &SpringBone3D::set_damping);
	ClassDB::bind_method(D_METHOD("get_damping"), &SpringBone3D::get_damping);

	ClassDB::bind_method(D_METHOD("set_additional_force", "additional_force"), &SpringBone3D::set_additional_force);
	ClassDB::bind_method(D_METHOD("get_additional_force"), &SpringBone3D::get_additional_force);

	ClassDB::bind_method(D_METHOD("set_influence", "influence"), &SpringBone3D::set_influence);
	ClassDB::bind_method(D_METHOD("get_influence"), &SpringBone3D::get_influence);

	ClassDB::bind_method(D_METHOD("set_stretchable", "stretchable"), &SpringBone3D::set_stretchable);
	ClassDB::bind_method(D_METHOD("is_stretchable"), &SpringBone3D::is_stretchable);

	ClassDB::bind_method(D_METHOD("set_tail_bone", "tail_bone"), &SpringBone3D::set_tail_bone);
	ClassDB::bind_method(D_METHOD("get_tail_bone"), &SpringBone3D::get_tail_bone);

	ClassDB::bind_method(D_METHOD("set_tail_position_offset", "tail_position_offset"), &SpringBone3D::set_tail_position_offset);
	ClassDB::bind_method(D_METHOD("get_tail_position_offset"), &SpringBone3D::get_tail_position_offset);

	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "enabled"), "set_enabled", "is_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::NODE_PATH, "skeleton_node", PROPERTY_HINT_NODE_PATH_VALID_TYPES, "Skeleton3D"), "set_skeleton_node", "get_skeleton_node");
	ADD_PROPERTY(PropertyInfo(Variant::STRING_NAME, "bone"), "set_bone", "get_bone");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "stiffness", PROPERTY_HINT_RANGE, "0.1,1,0.001"), "set_stiffness", "get_stiffness");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "damping", PROPERTY_HINT_RANGE, "0,1,0.001"), "set_damping", "get_damping");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "additional_force", PROPERTY_HINT_NONE, "suffix:m"), "set_additional_force", "get_additional_force");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "influence", PROPERTY_HINT_RANGE, "0,1,0.001"), "set_influence", "get_influence");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "stretchable"), "set_stretchable", "is_stretchable");
	ADD_PROPERTY(PropertyInfo(Variant::STRING_NAME, "tail_bone"), "set_tail_bone", "get_tail_bone");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "tail_position_offset", PROPERTY_HINT_NONE, "suffix:m"), "set_tail_position_offset", "get_tail_position_offset");
}

void SpringBone3D::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE: {
			skeleton_ref = Object::cast_to<Skeleton3D>(get_node_or_null(skeleton_node));
			reload_bone();
			set_physics_process_internal(enabled);
		} break;

		case NOTIFICATION_INTERNAL_PHYSICS_PROCESS: {
			process_spring(get_process_delta_time());
		} break;

		case NOTIFICATION_EXIT_TREE: {
			set_physics_process_internal(false);
		} break;
	}
}

Vector3 SpringBone3D::_adjust_tail_position(const Vector3 &p_tail_position, const Vector3 &p_previous_tail_position, const Vector3 &p_bone_position, real_t p_tail_length) {
	Vector3 new_pos;
	if (GDVIRTUAL_CALL(_adjust_tail_position, p_tail_position, p_previous_tail_position, p_bone_position, p_tail_length, new_pos)) {
		return new_pos;
	}
	return p_tail_position;
}

SpringBone3D::SpringBone3D() {
	skeleton_node = SceneStringNames::get_singleton()->path_pp;
}

SpringBone3D::~SpringBone3D() {
}

void SpringBone3D::set_enabled(bool p_enabled) {
	enabled = p_enabled;
	if (enabled) {
		reload_bone();
	} else {
		Skeleton3D *skeleton = get_skeleton();
		if (skeleton) {
			skeleton->clear_bones_global_pose_override();
		}
	}
	set_physics_process_internal(enabled);
}

bool SpringBone3D::is_enabled() const {
	return enabled;
}

void SpringBone3D::set_skeleton_node(const NodePath &p_path) {
	skeleton_node = p_path;
	skeleton_ref = Object::cast_to<Skeleton3D>(get_node_or_null(skeleton_node));
	reload_bone();
	notify_property_list_changed();
}

NodePath SpringBone3D::get_skeleton_node() const {
	return skeleton_node;
}

void SpringBone3D::set_bone(const StringName &p_bone) {
	if (bone != p_bone) {
		bone = p_bone;
		tail_bone = "";
		reload_bone();
		notify_property_list_changed();
	}
}

StringName SpringBone3D::get_bone() const {
	return bone;
}

void SpringBone3D::set_stiffness(real_t p_stiffness) {
	stiffness = p_stiffness;
}

real_t SpringBone3D::get_stiffness() const {
	return stiffness;
}

void SpringBone3D::set_damping(real_t p_damping) {
	damping = p_damping;
}

real_t SpringBone3D::get_damping() const {
	return damping;
}

void SpringBone3D::set_additional_force(const Vector3 &p_additional_force) {
	additional_force = p_additional_force;
}

const Vector3 &SpringBone3D::get_additional_force() const {
	return additional_force;
}

void SpringBone3D::set_influence(real_t p_influence) {
	influence = p_influence;
}

real_t SpringBone3D::get_influence() const {
	return influence;
}

void SpringBone3D::set_stretchable(bool p_stretchable) {
	stretchable = p_stretchable;
}

bool SpringBone3D::is_stretchable() const {
	return stretchable;
}

void SpringBone3D::set_tail_bone(const StringName &p_tail_bone) {
	tail_bone = p_tail_bone;
	reload_bone();
}

StringName SpringBone3D::get_tail_bone() const {
	return tail_bone;
}

void SpringBone3D::set_tail_position_offset(const Vector3 &p_tail_position_offset) {
	tail_position_offset = p_tail_position_offset;
	reload_bone();
}

const Vector3 &SpringBone3D::get_tail_position_offset() const {
	return tail_position_offset;
}

Skeleton3D *SpringBone3D::get_skeleton() const {
	return cast_to<Skeleton3D>(skeleton_ref.get_validated_object());
}

void SpringBone3D::reload_bone() {
	Skeleton3D *skeleton = get_skeleton();
	if (!skeleton) {
		return;
	}
	bone_id = skeleton->find_bone(bone);
	if (bone_id < 0) {
		return;
	}
	bone_id_parent = skeleton->get_bone_parent(bone_id);
	for (int id : skeleton->get_bone_children(bone_id)) {
		const String &child_bone_name = skeleton->get_bone_name(id);
		if (child_bone_name == tail_bone) {
			bone_id_tail = id;
			break;
		} else if (tail_bone == "") {
			tail_bone = child_bone_name;
			bone_id_tail = id;
			break;
		}
	}
	Transform3D skeleton_global_transf = skeleton->get_global_transform();
	Transform3D bone_transf_rest = skeleton->get_bone_global_rest(bone_id);
	Transform3D parent_bone_transf_rest = Transform3D();
	if (bone_id_parent >= 0) {
		parent_bone_transf_rest = skeleton->get_bone_global_rest(bone_id_parent);
	}
	Vector3 tail_loc_pos = tail_position_offset;
	if (bone_id_tail >= 0) {
		tail_loc_pos += skeleton->get_bone_rest(bone_id_tail).origin;
	} else if (bone_id_parent >= 0) {
		Vector3 off = skeleton->get_bone_rest(bone_id).origin;
		off = parent_bone_transf_rest.basis.xform(off);
		off = bone_transf_rest.basis.xform_inv(off);
		tail_loc_pos += off;
	}
	tail_pos = skeleton_global_transf.xform(bone_transf_rest.xform(tail_loc_pos));
	prev_pos = tail_pos;

	tail_dir = bone_transf_rest.basis.xform(tail_loc_pos);
	tail_dir = parent_bone_transf_rest.basis.xform_inv(tail_dir);

	skeleton->clear_bones_global_pose_override();
}

void SpringBone3D::process_spring(real_t p_delta) {
	Skeleton3D *skeleton = get_skeleton();
	if (!skeleton || bone_id < 0) {
		return;
	}
	Transform3D skeleton_global_transf = skeleton->get_global_transform();

	Transform3D bone_transf = skeleton->get_bone_global_pose(bone_id);
	Transform3D bone_transf_world = skeleton_global_transf * bone_transf;

	Transform3D bone_transf_rest_local = skeleton->get_bone_rest(bone_id);
	Transform3D parent_transf = bone_id_parent >= 0 ? skeleton->get_bone_global_pose(bone_id_parent) : Transform3D();
	Transform3D parent_transf_world = skeleton_global_transf * parent_transf;
	Transform3D bone_transf_rest = parent_transf * bone_transf_rest_local;

	Vector3 tail_dir_world = parent_transf_world.basis.xform(tail_dir);

	real_t tail_len = tail_dir_world.length();
	tail_dir_world = tail_dir_world / tail_len;

	Vector3 tail_current = tail_pos;
	Vector3 inertia = (tail_current - prev_pos) * (1 - damping);
	Vector3 stiff = tail_dir_world * stiffness;

	Vector3 new_tail = tail_current + inertia + p_delta * (stiff + additional_force);
	Vector3 to_tail = (new_tail - bone_transf_world.origin).normalized();

	// Keep tail length to make sure it's not stretched.
	if (!stretchable) {
		new_tail = bone_transf_world.origin + to_tail * tail_len;
	}

	new_tail = _adjust_tail_position(new_tail, tail_current, bone_transf_world.origin, tail_len);

	prev_pos = tail_current;
	tail_pos = new_tail;

	// Rotate to new tail direction.
	Vector3 bone_rotate_axis = tail_dir_world.cross(to_tail);
	real_t axis_len = bone_rotate_axis.length();
	real_t dot = tail_dir_world.dot(to_tail);
	real_t bone_rotate_angle = Math::acos(dot);

	if (Math::is_zero_approx(axis_len) || isnan(bone_rotate_angle)) {
		return;
	}

	bone_rotate_axis = bone_rotate_axis / axis_len;
	bone_rotate_axis = skeleton_global_transf.basis.xform_inv(bone_rotate_axis).normalized();
	Basis new_basis = bone_transf_rest.basis.rotated(bone_rotate_axis, bone_rotate_angle);
	Transform3D bone_new_transf_obj = Transform3D(new_basis, bone_transf.origin);

	skeleton->set_bone_global_pose_override(bone_id, bone_new_transf_obj, MIN(influence, 1.0), true);
}

#endif // _3D_DISABLED
