/*************************************************************************/
/*  skeleton_modification_3d.cpp                                         */
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

#include "skeleton_modification_3d.h"
#include "scene/3d/skeleton_3d.h"

void SkeletonModification3D::_validate_property(PropertyInfo &p_property) const {
	if (is_property_hidden(p_property.name)) {
		p_property.usage = PROPERTY_USAGE_NO_EDITOR;
	}
	if (is_bone_property(p_property.name)) {
		// Because it is a constant function, we cannot use the _get_skeleton_3d function.
		const Skeleton3D *skel = get_skeleton();

		if (skel) {
			p_property.hint = PROPERTY_HINT_ENUM;
			p_property.hint_string = get_bone_name_list();
		} else {
			p_property.hint = PROPERTY_HINT_NONE;
			p_property.hint_string = "";
		}
	}
}

void SkeletonModification3D::set_enabled(bool p_enabled) {
	enabled = p_enabled;
	if (is_inside_tree()) {
		set_process_internal(enabled && (!Engine::get_singleton()->is_editor_hint() || run_in_editor));
	}
}

bool SkeletonModification3D::get_enabled() const {
	return enabled;
}

void SkeletonModification3D::set_run_in_editor(bool p_enabled_in_editor) {
	run_in_editor = p_enabled_in_editor;
	if (Engine::get_singleton()->is_editor_hint() && is_inside_tree()) {
		set_process_internal(enabled && run_in_editor);
	}
}

bool SkeletonModification3D::get_run_in_editor() const {
	return run_in_editor;
}

String SkeletonModification3D::get_bone_name_list() const {
	if (bone_name_list.is_empty()) {
		const Skeleton3D *skel = get_skeleton();
		if (!skel) {
			return String();
		}
		for (int i = 0; i < skel->get_bone_count(); i++) {
			if (i > 0) {
				bone_name_list += ",";
			}
			bone_name_list += skel->get_bone_name(i);
		}
	}
	return bone_name_list;
}

void SkeletonModification3D::_bind_methods() {
	GDVIRTUAL_BIND(_execute, "delta");
	GDVIRTUAL_BIND(_skeleton_changed, "skeleton");
	GDVIRTUAL_BIND(_is_bone_property, "property_name");
	GDVIRTUAL_BIND(_is_property_hidden, "property_name");

	ClassDB::bind_method(D_METHOD("set_skeleton_path", "path"), &SkeletonModification3D::set_skeleton_path);
	ClassDB::bind_method(D_METHOD("get_skeleton_path"), &SkeletonModification3D::get_skeleton_path);
	ClassDB::bind_method(D_METHOD("set_enabled", "enabled"), &SkeletonModification3D::set_enabled);
	ClassDB::bind_method(D_METHOD("get_enabled"), &SkeletonModification3D::get_enabled);
	ClassDB::bind_method(D_METHOD("set_run_in_editor", "enabled_in_editor"), &SkeletonModification3D::set_run_in_editor);
	ClassDB::bind_method(D_METHOD("get_run_in_editor"), &SkeletonModification3D::get_run_in_editor);
	ClassDB::bind_method(D_METHOD("execute", "delta"), &SkeletonModification3D::execute);
	ClassDB::bind_method(D_METHOD("get_bone_name_list"), &SkeletonModification3D::get_bone_name_list);
	ClassDB::bind_method(D_METHOD("resolve_bone", "target_bone_name"), &SkeletonModification3D::resolve_bone);
	ClassDB::bind_method(D_METHOD("resolve_target", "target_node_path", "target_bone_name"), &SkeletonModification3D::resolve_target);
	ClassDB::bind_method(D_METHOD("get_target_transform", "resolved_target"), &SkeletonModification3D::get_target_transform);
	ClassDB::bind_method(D_METHOD("get_target_quaternion", "resolved_target"), &SkeletonModification3D::get_target_quaternion);
	ClassDB::bind_method(D_METHOD("get_bone_rest_forward_vector", "bone_idx"), &SkeletonModification3D::get_bone_rest_forward_vector);
	ClassDB::bind_method(D_METHOD("global_pose_z_forward_to_bone_forward", "bone_forward_vector", "basis"), &SkeletonModification3D::global_pose_z_forward_to_bone_forward);

	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "enabled"), "set_enabled", "get_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "run_in_editor"), "set_run_in_editor", "get_run_in_editor");
	ADD_PROPERTY(PropertyInfo(Variant::NODE_PATH, "skeleton_path"), "set_skeleton_path", "get_skeleton_path");
}

NodePath SkeletonModification3D::get_skeleton_path() const {
	return skeleton_path;
}

void SkeletonModification3D::set_skeleton_path(NodePath p_path) {
	if (p_path.is_empty()) {
		p_path = NodePath("..");
	}
	skeleton_path = p_path;
	skeleton_change_queued = true;
	cached_skeleton = Variant();
	bone_name_list.clear();
	update_configuration_warnings();
}

Skeleton3D *SkeletonModification3D::get_skeleton() const {
	Skeleton3D *skeleton_node = cast_to<Skeleton3D>(cached_skeleton);
	if (skeleton_node == nullptr) {
		skeleton_node = cast_to<Skeleton3D>(get_node_or_null(skeleton_path));
		cached_skeleton = skeleton_node;
	}
	return skeleton_node;
}

void SkeletonModification3D::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE: {
			set_process_internal(enabled && (!Engine::get_singleton()->is_editor_hint() || run_in_editor));
			cached_skeleton = Variant();
			if (Engine::get_singleton()->is_editor_hint()) {
				call_deferred(SNAME("update_configuration_warnings"));
			}
		} break;
		case NOTIFICATION_READY: {
			Skeleton3D *skel = get_skeleton();
			if (skel) {
				skeleton_changed(skel);
			}
		} break;
		case NOTIFICATION_INTERNAL_PROCESS: {
			ERR_FAIL_COND(!enabled);
			execute(get_process_delta_time());
		} break;
	}
}

void SkeletonModification3D::skeleton_changed(Skeleton3D *skeleton) {
	bone_name_list.clear();
	cached_skeleton_version = skeleton->get_version();
	skeleton_change_queued = false;
	GDVIRTUAL_CALL(_skeleton_changed, skeleton);
}

int SkeletonModification3D::resolve_bone(const String &target_bone_name) const {
	Skeleton3D *skel = get_skeleton();
	if (skel) {
		return skel->find_bone(target_bone_name);
	}
	return -1;
}

Variant SkeletonModification3D::resolve_target(const NodePath &target_node_path, const String &target_bone_name) const {
	if (target_node_path.is_empty()) {
		Skeleton3D *skel = get_skeleton();
		if (skel) {
			int found_bone = skel->find_bone(target_bone_name);
			if (found_bone >= 0) {
				return Variant(found_bone);
			}
		}
	} else {
		Node *resolved_node = get_node(target_node_path);
		if (cast_to<Node3D>(resolved_node)) {
			return Variant(resolved_node);
		}
	}
	return Variant(false);
}

Transform3D SkeletonModification3D::get_target_transform(Variant resolved_target) const {
	if (resolved_target.get_type() == Variant::OBJECT) {
		Skeleton3D *skel = get_skeleton();
		Node3D *resolved_node3d = cast_to<Node3D>((Object *)resolved_target);
		return skel->get_global_transform().affine_inverse() * resolved_node3d->get_global_transform();
	} else if (resolved_target.get_type() == Variant::INT) {
		int resolved_bone = (int)resolved_target;
		return get_bone_transform(resolved_bone);
	}
	ERR_FAIL_V_MSG(Transform3D(), "Looking up transform of unresolved target.");
}

Quaternion SkeletonModification3D::get_target_quaternion(Variant resolved_target) const {
	if (resolved_target.get_type() == Variant::OBJECT) {
		Skeleton3D *skel = get_skeleton();
		Node3D *resolved_node3d = cast_to<Node3D>((Object *)resolved_target);
		return skel->get_global_transform().basis.get_rotation_quaternion().inverse() * resolved_node3d->get_global_transform().basis.get_rotation_quaternion();
	} else if (resolved_target.get_type() == Variant::INT) {
		int resolved_bone = (int)resolved_target;
		return get_bone_quaternion(resolved_bone);
	}
	ERR_FAIL_V_MSG(Quaternion(), "Looking up quaternion of unresolved target.");
}

Transform3D SkeletonModification3D::get_bone_transform(int resolved_bone) const {
	Skeleton3D *skel = get_skeleton();
	ERR_FAIL_COND_V(resolved_bone < 0, Transform3D());
	Transform3D xform = skel->get_bone_pose(resolved_bone);
	resolved_bone = skel->get_bone_parent(resolved_bone);
	while (resolved_bone >= 0) {
		xform = skel->get_bone_pose(resolved_bone) * xform;
		resolved_bone = skel->get_bone_parent(resolved_bone);
	}
	return xform;
}

Quaternion SkeletonModification3D::get_bone_quaternion(int resolved_bone) const {
	Skeleton3D *skel = get_skeleton();
	ERR_FAIL_COND_V(resolved_bone < 0, Quaternion());
	Quaternion quat = skel->get_bone_pose_rotation(resolved_bone);
	resolved_bone = skel->get_bone_parent(resolved_bone);
	while (resolved_bone >= 0) {
		quat = skel->get_bone_pose_rotation(resolved_bone) * quat;
		resolved_bone = skel->get_bone_parent(resolved_bone);
	}
	return quat;
}

// The forward direction vector and rest bone forward axis should be cached because they do
// not change 99% of the time, but recalculating them can be expensive on models with many bones.
Vector3 SkeletonModification3D::get_bone_rest_forward_vector(int p_bone) {
	Skeleton3D *skeleton = get_skeleton();
	ERR_FAIL_COND_V(skeleton == nullptr, Vector3(0, 1, 0));

	Transform3D rest = skeleton->get_bone_rest(p_bone);

	// If it is a child/leaf bone...
	if (skeleton->get_bone_parent(p_bone) > 0) {
		return rest.origin.normalized();
	} else {
		// If it has children...
		Vector<int> child_bones = skeleton->get_bone_children(p_bone);
		if (child_bones.size() > 0) {
			Vector3 combined_child_dir = Vector3(0, 0, 0);
			for (int i = 0; i < child_bones.size(); i++) {
				combined_child_dir += skeleton->get_bone_rest(child_bones[i]).origin.normalized();
			}
			combined_child_dir = combined_child_dir / child_bones.size();
			return combined_child_dir.normalized();
		} else {
			WARN_PRINT_ONCE("Cannot calculate forward direction for bone " + itos(p_bone));
			WARN_PRINT_ONCE("Assuming direction of (0, 1, 0) for bone");
			return Vector3(0, 1, 0);
		}
	}
}

SkeletonModification3D::Bone_Forward_Axis SkeletonModification3D::vector_to_forward_axis(Vector3 p_rest_bone_forward_vector) {
	Vector3 forward_axis_absolute = p_rest_bone_forward_vector.abs();
	if (forward_axis_absolute.x > forward_axis_absolute.y && forward_axis_absolute.x > forward_axis_absolute.z) {
		return (p_rest_bone_forward_vector.x > 0) ? BONE_AXIS_X_FORWARD : BONE_AXIS_NEGATIVE_X_FORWARD;
	} else if (forward_axis_absolute.y > forward_axis_absolute.x && forward_axis_absolute.y > forward_axis_absolute.z) {
		return (p_rest_bone_forward_vector.y > 0) ? BONE_AXIS_Y_FORWARD : BONE_AXIS_NEGATIVE_Y_FORWARD;
	} else {
		return (p_rest_bone_forward_vector.z > 0) ? BONE_AXIS_Z_FORWARD : BONE_AXIS_NEGATIVE_Z_FORWARD;
	}
}

Basis SkeletonModification3D::global_pose_z_forward_to_bone_forward(Vector3 p_bone_forward_vector, Basis p_basis) {
	Basis return_basis = p_basis;

	Bone_Forward_Axis axis = vector_to_forward_axis(p_bone_forward_vector);
	switch (axis) {
		case BONE_AXIS_X_FORWARD:
			return_basis.rotate_local(Vector3(0, 1, 0), (Math_PI / 2.0));
			break;
		case BONE_AXIS_NEGATIVE_X_FORWARD:
			return_basis.rotate_local(Vector3(0, 1, 0), -(Math_PI / 2.0));
			break;
		case BONE_AXIS_Y_FORWARD:
			return_basis.rotate_local(Vector3(1, 0, 0), -(Math_PI / 2.0));
			break;
		case BONE_AXIS_NEGATIVE_Y_FORWARD:
			return_basis.rotate_local(Vector3(1, 0, 0), (Math_PI / 2.0));
			break;
		case BONE_AXIS_Z_FORWARD:
			// Do nothing!
			break;
		case BONE_AXIS_NEGATIVE_Z_FORWARD:
			return_basis.rotate_local(Vector3(0, 0, 1), Math_PI);
			break;
	}
	return return_basis;
}

void SkeletonModification3D::execute(real_t delta) {
	Skeleton3D *skel = get_skeleton();
	if (skel != nullptr) {
		if (skel->get_version() != cached_skeleton_version || skeleton_change_queued) {
			skeleton_changed(skel);
		}
	}
	GDVIRTUAL_CALL(_execute, delta);
}

bool SkeletonModification3D::is_property_hidden(String p_property_name) const {
	bool ret = false;
	const_cast<SkeletonModification3D *>(this)->GDVIRTUAL_CALL(_is_property_hidden, p_property_name, ret);
	return ret;
}

bool SkeletonModification3D::is_bone_property(String p_property_name) const {
	bool ret = false;
	const_cast<SkeletonModification3D *>(this)->GDVIRTUAL_CALL(_is_bone_property, p_property_name, ret);
	return ret;
}

PackedStringArray SkeletonModification3D::get_configuration_warnings() const {
	PackedStringArray ret = Node::get_configuration_warnings();

	if (!get_skeleton()) {
		ret.push_back("Modification skeleton_path must point to a Skeleton3D node.");
	}

	return ret;
}
