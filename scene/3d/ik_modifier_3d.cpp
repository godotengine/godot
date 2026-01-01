/**************************************************************************/
/*  ik_modifier_3d.cpp                                                    */
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

#include "ik_modifier_3d.h"

void IKModifier3D::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE: {
#ifdef TOOLS_ENABLED
			if (Engine::get_singleton()->is_editor_hint()) {
				set_notify_local_transform(true); // Used for updating gizmo in editor.
			}
			_update_mutable_info();
			_make_gizmo_dirty();
#endif // TOOLS_ENABLED
			_make_all_joints_dirty();
		} break;
#ifdef TOOLS_ENABLED
		case NOTIFICATION_LOCAL_TRANSFORM_CHANGED: {
			_make_gizmo_dirty();
		} break;
#endif // TOOLS_ENABLED
	}
}

void IKModifier3D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_setting_count", "count"), &IKModifier3D::set_setting_count);
	ClassDB::bind_method(D_METHOD("get_setting_count"), &IKModifier3D::get_setting_count);
	ClassDB::bind_method(D_METHOD("clear_settings"), &IKModifier3D::clear_settings);

	ClassDB::bind_method(D_METHOD("set_mutable_bone_axes", "enabled"), &IKModifier3D::set_mutable_bone_axes);
	ClassDB::bind_method(D_METHOD("are_bone_axes_mutable"), &IKModifier3D::are_bone_axes_mutable);

	// To process manually.
	ClassDB::bind_method(D_METHOD("reset"), &IKModifier3D::reset);

	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "mutable_bone_axes"), "set_mutable_bone_axes", "are_bone_axes_mutable");
}

void IKModifier3D::_set_active(bool p_active) {
	if (p_active) {
		reset();
	}
}

void IKModifier3D::_skeleton_changed(Skeleton3D *p_old, Skeleton3D *p_new) {
	if (p_old && p_old->is_connected(SNAME("rest_updated"), callable_mp(this, &IKModifier3D::_rest_updated))) {
		p_old->disconnect(SNAME("rest_updated"), callable_mp(this, &IKModifier3D::_rest_updated));
	}
	if (p_new && !p_new->is_connected(SNAME("rest_updated"), callable_mp(this, &IKModifier3D::_rest_updated))) {
		p_new->connect(SNAME("rest_updated"), callable_mp(this, &IKModifier3D::_rest_updated));
	}
	_make_all_joints_dirty();
}

void IKModifier3D::_validate_bone_names() {
	//
}

void IKModifier3D::_rest_updated() {
	_make_all_joints_dirty();
	if (is_inside_tree()) {
		Skeleton3D *skeleton = get_skeleton();
		if (skeleton) {
			for (uint32_t i = 0; i < settings.size(); i++) {
				_init_joints(skeleton, i);
			}
		}
	}
#ifdef TOOLS_ENABLED
	_update_mutable_info();
	_make_gizmo_dirty();
#endif // TOOLS_ENABLED
}

void IKModifier3D::_make_all_joints_dirty() {
	//
}

void IKModifier3D::_init_joints(Skeleton3D *p_skeleton, int p_index) {
	//
}

void IKModifier3D::_update_joints(int p_index) {
	//
}

void IKModifier3D::_make_simulation_dirty(int p_index) {
	//
}

void IKModifier3D::_process_modification(double p_delta) {
	Skeleton3D *skeleton = get_skeleton();
	if (!skeleton) {
		return;
	}

	_process_ik(skeleton, p_delta);
}

void IKModifier3D::_process_ik(Skeleton3D *p_skeleton, double p_delta) {
	//
}

void IKModifier3D::_update_bone_axis(Skeleton3D *p_skeleton, int p_index) {
	//
}

#ifdef TOOLS_ENABLED
void IKModifier3D::_make_gizmo_dirty() {
	if (gizmo_dirty) {
		return;
	}
	gizmo_dirty = true;

	callable_mp(this, &IKModifier3D::_redraw_gizmo).call_deferred();
}

void IKModifier3D::_update_mutable_info() {
	//
}

void IKModifier3D::_redraw_gizmo() {
	update_gizmos();
	gizmo_dirty = false;
}
#endif // TOOLS_ENABLED

void IKModifier3D::set_mutable_bone_axes(bool p_enabled) {
	mutable_bone_axes = p_enabled;
	for (uint32_t i = 0; i < settings.size(); i++) {
		_make_simulation_dirty(i);
	}
#ifdef TOOLS_ENABLED
	_update_mutable_info();
#endif // TOOLS_ENABLED
}

bool IKModifier3D::are_bone_axes_mutable() const {
	return mutable_bone_axes;
}

Quaternion IKModifier3D::get_local_pose_rotation(Skeleton3D *p_skeleton, int p_bone, const Quaternion &p_global_pose_rotation) {
	int parent = p_skeleton->get_bone_parent(p_bone);
	if (parent < 0) {
		return p_global_pose_rotation;
	}
	return p_skeleton->get_bone_global_pose(parent).basis.get_rotation_quaternion().inverse() * p_global_pose_rotation;
}

Vector3 IKModifier3D::get_bone_axis(Skeleton3D *p_skeleton, int p_end_bone, BoneDirection p_direction, bool p_mutable_bone_axes) {
	if (!p_skeleton->is_inside_tree()) {
		return Vector3();
	}
	Vector3 axis;
	if (p_direction == BONE_DIRECTION_FROM_PARENT) {
		if (p_skeleton) {
			axis = p_skeleton->get_bone_rest(p_end_bone).basis.xform_inv(p_mutable_bone_axes ? p_skeleton->get_bone_pose(p_end_bone).origin : p_skeleton->get_bone_rest(p_end_bone).origin);
			axis.normalize();
		}
	} else {
		axis = get_vector_from_bone_axis(static_cast<BoneAxis>((int)p_direction));
	}
	return axis;
}

int IKModifier3D::get_setting_count() const {
	return settings.size();
}

void IKModifier3D::reset() {
	Skeleton3D *skeleton = get_skeleton();
	if (!skeleton) {
		return;
	}
	for (uint32_t i = 0; i < settings.size(); i++) {
		_make_simulation_dirty(i);
		_init_joints(skeleton, i);
	}
}

IKModifier3D::~IKModifier3D() {
	clear_settings();
}
