/**************************************************************************/
/*  skeleton_modifier_3d.cpp                                              */
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

#include "skeleton_modifier_3d.h"

void SkeletonModifier3D::_validate_property(PropertyInfo &p_property) const {
	//
}

PackedStringArray SkeletonModifier3D::get_configuration_warnings() const {
	PackedStringArray warnings = Node3D::get_configuration_warnings();
	if (skeleton_id.is_null()) {
		warnings.push_back(RTR("Skeleton3D node not set! SkeletonModifier3D must be child of Skeleton3D or set a path to an external skeleton."));
	}
	return warnings;
}

/* Skeleton3D */

Skeleton3D *SkeletonModifier3D::get_skeleton() const {
	return Object::cast_to<Skeleton3D>(ObjectDB::get_instance(skeleton_id));
}

void SkeletonModifier3D::_update_skeleton_path() {
	skeleton_id = ObjectID();

	// Make sure parent is a Skeleton3D.
	Skeleton3D *sk = Object::cast_to<Skeleton3D>(get_parent());
	if (sk) {
		skeleton_id = sk->get_instance_id();
	}
}

void SkeletonModifier3D::_update_skeleton() {
	if (!is_inside_tree()) {
		return;
	}
	Skeleton3D *old_sk = get_skeleton();
	_update_skeleton_path();
	Skeleton3D *new_sk = get_skeleton();
	if (old_sk != new_sk) {
		_skeleton_changed(old_sk, new_sk);
	}
	update_configuration_warnings();
}

void SkeletonModifier3D::_skeleton_changed(Skeleton3D *p_old, Skeleton3D *p_new) {
	//
}

/* Process */

void SkeletonModifier3D::set_active(bool p_active) {
	if (active == p_active) {
		return;
	}
	active = p_active;
	_set_active(active);
}

bool SkeletonModifier3D::is_active() const {
	return active;
}

void SkeletonModifier3D::_set_active(bool p_active) {
	//
}

void SkeletonModifier3D::set_influence(real_t p_influence) {
	influence = p_influence;
}

real_t SkeletonModifier3D::get_influence() const {
	return influence;
}

void SkeletonModifier3D::process_modification() {
	if (!active) {
		return;
	}
	_process_modification();
	emit_signal(SNAME("modification_processed"));
}

void SkeletonModifier3D::_process_modification() {
	GDVIRTUAL_CALL(_process_modification);
}

void SkeletonModifier3D::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE:
		case NOTIFICATION_PARENTED: {
			_update_skeleton();
		} break;
	}
}

void SkeletonModifier3D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_active", "active"), &SkeletonModifier3D::set_active);
	ClassDB::bind_method(D_METHOD("is_active"), &SkeletonModifier3D::is_active);

	ClassDB::bind_method(D_METHOD("set_influence", "influence"), &SkeletonModifier3D::set_influence);
	ClassDB::bind_method(D_METHOD("get_influence"), &SkeletonModifier3D::get_influence);

	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "active"), "set_active", "is_active");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "influence", PROPERTY_HINT_RANGE, "0,1,0.001"), "set_influence", "get_influence");

	ADD_SIGNAL(MethodInfo("modification_processed"));
	GDVIRTUAL_BIND(_process_modification);
}

SkeletonModifier3D::SkeletonModifier3D() {
}
