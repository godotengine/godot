/**************************************************************************/
/*  skeleton_2d.cpp                                                       */
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

#include "skeleton_2d.h"
#include "core/engine.h"
#include "core/math/transform_interpolator.h"

void Bone2D::_order_changed_in_parent() {
	if (skeleton) {
		skeleton->_make_bone_setup_dirty();
	}
}

void Bone2D::_notification(int p_what) {
	if (p_what == NOTIFICATION_ENTER_TREE) {
		Node *parent = get_parent();

		if (parent) {
			parent->connect("child_order_changed", this, "_order_changed_in_parent");
		}

		parent_bone = Object::cast_to<Bone2D>(parent);
		skeleton = nullptr;
		while (parent) {
			skeleton = Object::cast_to<Skeleton2D>(parent);
			if (skeleton) {
				break;
			}
			if (!Object::cast_to<Bone2D>(parent)) {
				break; //skeletons must be chained to Bone2Ds.
			}

			parent = parent->get_parent();
		}

		if (skeleton) {
			Skeleton2D::Bone bone;
			bone.bone = this;
			skeleton->bones.push_back(bone);
			skeleton->_make_bone_setup_dirty();
		}
	}
	if (p_what == NOTIFICATION_LOCAL_TRANSFORM_CHANGED) {
		if (skeleton) {
			skeleton->_make_transform_dirty();
		}
	}

	if (p_what == NOTIFICATION_EXIT_TREE) {
		Node *parent = get_parent();
		if (parent) {
			parent->disconnect("child_order_changed", this, "_order_changed_in_parent");
		}

		if (skeleton) {
			for (int i = 0; i < skeleton->bones.size(); i++) {
				if (skeleton->bones[i].bone == this) {
					skeleton->bones.remove(i);
					break;
				}
			}
			skeleton->_make_bone_setup_dirty();
			skeleton = nullptr;
		}
		parent_bone = nullptr;
	}
}
void Bone2D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_rest", "rest"), &Bone2D::set_rest);
	ClassDB::bind_method(D_METHOD("get_rest"), &Bone2D::get_rest);
	ClassDB::bind_method(D_METHOD("apply_rest"), &Bone2D::apply_rest);
	ClassDB::bind_method(D_METHOD("get_skeleton_rest"), &Bone2D::get_skeleton_rest);
	ClassDB::bind_method(D_METHOD("get_index_in_skeleton"), &Bone2D::get_index_in_skeleton);

	ClassDB::bind_method(D_METHOD("set_default_length", "default_length"), &Bone2D::set_default_length);
	ClassDB::bind_method(D_METHOD("get_default_length"), &Bone2D::get_default_length);

	ClassDB::bind_method(D_METHOD("_order_changed_in_parent"), &Bone2D::_order_changed_in_parent);

	ADD_PROPERTY(PropertyInfo(Variant::TRANSFORM2D, "rest"), "set_rest", "get_rest");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "default_length", PROPERTY_HINT_RANGE, "1,1024,1"), "set_default_length", "get_default_length");
}

void Bone2D::set_rest(const Transform2D &p_rest) {
	rest = p_rest;
	if (skeleton) {
		skeleton->_make_bone_setup_dirty();
	}

	update_configuration_warning();
}

Transform2D Bone2D::get_rest() const {
	return rest;
}

Transform2D Bone2D::get_skeleton_rest() const {
	if (parent_bone) {
		return parent_bone->get_skeleton_rest() * rest;
	} else {
		return rest;
	}
}

void Bone2D::apply_rest() {
	set_transform(rest);
}

void Bone2D::set_default_length(float p_length) {
	default_length = p_length;
}

float Bone2D::get_default_length() const {
	return default_length;
}

int Bone2D::get_index_in_skeleton() const {
	ERR_FAIL_COND_V(!skeleton, -1);
	skeleton->_update_bone_setup();
	return skeleton_index;
}
String Bone2D::get_configuration_warning() const {
	String warning = Node2D::get_configuration_warning();
	if (!skeleton) {
		if (warning != String()) {
			warning += "\n\n";
		}
		if (parent_bone) {
			warning += TTR("This Bone2D chain should end at a Skeleton2D node.");
		} else {
			warning += TTR("A Bone2D only works with a Skeleton2D or another Bone2D as parent node.");
		}
	}

	if (rest == Transform2D(0, 0, 0, 0, 0, 0)) {
		if (warning != String()) {
			warning += "\n\n";
		}
		warning += TTR("This bone lacks a proper REST pose. Go to the Skeleton2D node and set one.");
	}

	return warning;
}

Bone2D::Bone2D() {
	skeleton = nullptr;
	parent_bone = nullptr;
	skeleton_index = -1;
	default_length = 16;
	set_notify_local_transform(true);
	//this is a clever hack so the bone knows no rest has been set yet, allowing to show an error.
	for (int i = 0; i < 3; i++) {
		rest[i] = Vector2(0, 0);
	}
}

//////////////////////////////////////

void Skeleton2D::_make_bone_setup_dirty() {
	if (bone_setup_dirty) {
		return;
	}
	bone_setup_dirty = true;
	if (is_inside_tree()) {
		call_deferred("_update_bone_setup");
	}
}

void Skeleton2D::_update_bone_setup() {
	if (!bone_setup_dirty) {
		return;
	}

	bone_setup_dirty = false;
	VS::get_singleton()->skeleton_allocate(skeleton, bones.size(), true);

	bones.sort(); //sorty so they are always in the same order/index

	for (int i = 0; i < bones.size(); i++) {
		bones.write[i].rest_inverse = bones[i].bone->get_skeleton_rest().affine_inverse(); //bind pose
		bones.write[i].bone->skeleton_index = i;
		Bone2D *parent_bone = Object::cast_to<Bone2D>(bones[i].bone->get_parent());
		if (parent_bone) {
			bones.write[i].parent_index = parent_bone->skeleton_index;
		} else {
			bones.write[i].parent_index = -1;
		}
	}

	transform_dirty = true;
	_update_transform();
	emit_signal("bone_setup_changed");
}

void Skeleton2D::_make_transform_dirty() {
	if (transform_dirty) {
		return;
	}
	transform_dirty = true;
	if (is_inside_tree()) {
		call_deferred("_update_transform");
	}
}

void Skeleton2D::_update_transform() {
	if (bone_setup_dirty) {
		_update_bone_setup();
		return; //above will update transform anyway
	}
	if (!transform_dirty) {
		return;
	}

	transform_dirty = false;

	for (int i = 0; i < bones.size(); i++) {
		ERR_CONTINUE(bones[i].parent_index >= i);
		if (bones[i].parent_index >= 0) {
			bones.write[i].accum_transform = bones[bones[i].parent_index].accum_transform * bones[i].bone->get_transform();
		} else {
			bones.write[i].accum_transform = bones[i].bone->get_transform();
		}
	}

	for (int i = 0; i < bones.size(); i++) {
		Transform2D final_xform = bones[i].accum_transform * bones[i].rest_inverse;
		VS::get_singleton()->skeleton_bone_set_transform_2d(skeleton, i, final_xform);
	}
}

int Skeleton2D::get_bone_count() const {
	ERR_FAIL_COND_V(!is_inside_tree(), 0);

	if (bone_setup_dirty) {
		const_cast<Skeleton2D *>(this)->_update_bone_setup();
	}

	return bones.size();
}

Bone2D *Skeleton2D::get_bone(int p_idx) {
	ERR_FAIL_COND_V(!is_inside_tree(), nullptr);
	ERR_FAIL_INDEX_V(p_idx, bones.size(), nullptr);

	return bones[p_idx].bone;
}

void Skeleton2D::_update_process_mode() {
	bool process = is_physics_interpolated_and_enabled() && is_visible_in_tree();

	set_process_internal(process);
	set_physics_process_internal(process);
}

void Skeleton2D::_ensure_update_interpolation_data() {
	uint64_t tick = Engine::get_singleton()->get_physics_frames();

	if (_interpolation_data.last_update_physics_tick != tick) {
		_interpolation_data.xform_prev = _interpolation_data.xform_curr;
		_interpolation_data.last_update_physics_tick = tick;
	}
}

void Skeleton2D::_physics_interpolated_changed() {
	_update_process_mode();
}

void Skeleton2D::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_READY: {
			if (bone_setup_dirty) {
				_update_bone_setup();
			}
			if (transform_dirty) {
				_update_transform();
			}

			request_ready();
		} break;
		case NOTIFICATION_ENTER_TREE: {
			_update_process_mode();

			if (is_physics_interpolated_and_enabled()) {
				_interpolation_data.xform_curr = get_global_transform();
				_interpolation_data.xform_prev = _interpolation_data.xform_curr;
			}
		} break;
		case NOTIFICATION_TRANSFORM_CHANGED: {
			if (is_physics_interpolated_and_enabled()) {
				_ensure_update_interpolation_data();
				if (Engine::get_singleton()->is_in_physics_frame()) {
					_interpolation_data.xform_curr = get_global_transform();
				}
			} else {
				VS::get_singleton()->skeleton_set_base_transform_2d(skeleton, get_global_transform());
			}
		} break;
		case NOTIFICATION_RESET_PHYSICS_INTERPOLATION: {
			_interpolation_data.xform_curr = get_global_transform();
			_interpolation_data.xform_prev = _interpolation_data.xform_curr;
		} break;
		case NOTIFICATION_INTERNAL_PHYSICS_PROCESS: {
			if (is_physics_interpolated_and_enabled()) {
				_ensure_update_interpolation_data();
				_interpolation_data.xform_curr = get_global_transform();
			}
		} break;
		case NOTIFICATION_INTERNAL_PROCESS: {
			if (is_physics_interpolated_and_enabled()) {
				Transform2D res;
				TransformInterpolator::interpolate_transform_2d(_interpolation_data.xform_prev, _interpolation_data.xform_curr, res, Engine::get_singleton()->get_physics_interpolation_fraction());
				VS::get_singleton()->skeleton_set_base_transform_2d(skeleton, res);
			}
		} break;
		case NOTIFICATION_VISIBILITY_CHANGED: {
			_update_process_mode();
		} break;
	}
}

RID Skeleton2D::get_skeleton() const {
	return skeleton;
}
void Skeleton2D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("_update_bone_setup"), &Skeleton2D::_update_bone_setup);
	ClassDB::bind_method(D_METHOD("_update_transform"), &Skeleton2D::_update_transform);

	ClassDB::bind_method(D_METHOD("get_bone_count"), &Skeleton2D::get_bone_count);
	ClassDB::bind_method(D_METHOD("get_bone", "idx"), &Skeleton2D::get_bone);

	ClassDB::bind_method(D_METHOD("get_skeleton"), &Skeleton2D::get_skeleton);

	ADD_SIGNAL(MethodInfo("bone_setup_changed"));
}

Skeleton2D::Skeleton2D() {
	bone_setup_dirty = true;
	transform_dirty = true;

	skeleton = RID_PRIME(VS::get_singleton()->skeleton_create());
	set_notify_transform(true);
}

Skeleton2D::~Skeleton2D() {
	VS::get_singleton()->free(skeleton);
}
