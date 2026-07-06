/**************************************************************************/
/*  skeleton_modifier_2d.cpp                                              */
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

#include "skeleton_modifier_2d.h"

#include "core/object/class_db.h"

PackedStringArray SkeletonModifier2D::get_configuration_warnings() const {
	PackedStringArray warnings = Node2D::get_configuration_warnings();
	if (skeleton_id.is_null()) {
		warnings.push_back(RTR("Skeleton2D node not set! SkeletonModifier2D must be child of Skeleton2D."));
	}
	return warnings;
}

Skeleton2D *SkeletonModifier2D::get_skeleton() const {
	return ObjectDB::get_instance<Skeleton2D>(skeleton_id);
}

void SkeletonModifier2D::_update_skeleton_path() {
	skeleton_id = ObjectID();

	Skeleton2D *sk = Object::cast_to<Skeleton2D>(get_parent());
	if (sk) {
		skeleton_id = sk->get_instance_id();
	}
}

void SkeletonModifier2D::_update_skeleton() {
	if (!is_inside_tree()) {
		return;
	}
	Skeleton2D *old_sk = get_skeleton();
	_update_skeleton_path();
	Skeleton2D *new_sk = get_skeleton();
	if (old_sk != new_sk) {
		_skeleton_changed(old_sk, new_sk);
	}
	if (new_sk) {
		_validate_bone_names();
	}
	update_configuration_warnings();
}

void SkeletonModifier2D::_skeleton_changed(Skeleton2D *p_old, Skeleton2D *p_new) {
	GDVIRTUAL_CALL(_skeleton_changed, p_old, p_new);
}

void SkeletonModifier2D::_validate_bone_names() {
	GDVIRTUAL_CALL(_validate_bone_names);
}

void SkeletonModifier2D::_force_update_skeleton() {
	if (!is_inside_tree()) {
		return;
	}
	Skeleton2D *skeleton = get_skeleton();
	if (!skeleton) {
		return;
	}
	skeleton->force_update_deferred();
}

void SkeletonModifier2D::set_active(bool p_active) {
	if (active == p_active) {
		return;
	}
	active = p_active;
	_set_active(active);
	_force_update_skeleton();
}

bool SkeletonModifier2D::is_active() const {
	return active;
}

void SkeletonModifier2D::_set_active(bool p_active) {
}

void SkeletonModifier2D::set_influence(real_t p_influence) {
	real_t new_influence = CLAMP(p_influence, 0.0, 1.0);
	if (Math::is_equal_approx(influence, new_influence)) {
		return;
	}
	influence = new_influence;
	_force_update_skeleton();
}

real_t SkeletonModifier2D::get_influence() const {
	return influence;
}

void SkeletonModifier2D::process_modification(double p_delta) {
	if (!is_inside_tree() || !active) {
		return;
	}
	_process_modification(p_delta);
	emit_signal(SNAME("modification_processed"));
}

void SkeletonModifier2D::_process_modification(double p_delta) {
	GDVIRTUAL_CALL(_process_modification, p_delta);
}

void SkeletonModifier2D::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE:
		case NOTIFICATION_PARENTED: {
			_update_skeleton();
		} break;
		case NOTIFICATION_EXIT_TREE:
		case NOTIFICATION_UNPARENTED: {
			_force_update_skeleton();
			skeleton_id = ObjectID();
			update_configuration_warnings();
		} break;
	}
}

void SkeletonModifier2D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_skeleton"), &SkeletonModifier2D::get_skeleton);

	ClassDB::bind_method(D_METHOD("set_active", "active"), &SkeletonModifier2D::set_active);
	ClassDB::bind_method(D_METHOD("is_active"), &SkeletonModifier2D::is_active);

	ClassDB::bind_method(D_METHOD("set_influence", "influence"), &SkeletonModifier2D::set_influence);
	ClassDB::bind_method(D_METHOD("get_influence"), &SkeletonModifier2D::get_influence);

	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "active"), "set_active", "is_active");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "influence", PROPERTY_HINT_RANGE, "0,1,0.001"), "set_influence", "get_influence");

	ADD_SIGNAL(MethodInfo("modification_processed"));

	GDVIRTUAL_BIND(_process_modification, "delta");
	GDVIRTUAL_BIND(_skeleton_changed, "old_skeleton", "new_skeleton");
	GDVIRTUAL_BIND(_validate_bone_names);
}

SkeletonModifier2D::SkeletonModifier2D() {
}
