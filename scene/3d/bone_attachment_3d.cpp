/*************************************************************************/
/*  bone_attachment_3d.cpp                                               */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2021 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2021 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "bone_attachment_3d.h"

void BoneAttachment3D::_validate_property(PropertyInfo &property) const {
	if (property.name == "bone_name") {
		Skeleton3D *parent = Object::cast_to<Skeleton3D>(get_parent());

		if (parent) {
			String names;
			for (int i = 0; i < parent->get_bone_count(); i++) {
				if (i > 0) {
					names += ",";
				}
				names += parent->get_bone_name(i);
			}

			property.hint = PROPERTY_HINT_ENUM;
			property.hint_string = names;
		} else {
			property.hint = PROPERTY_HINT_NONE;
			property.hint_string = "";
		}
	}
}

void BoneAttachment3D::_check_bind() {
	Skeleton3D *sk = Object::cast_to<Skeleton3D>(get_parent());
	if (sk && !bound) {
		if (bone_idx <= -1) {
			bone_idx = sk->find_bone(bone_name);
		}
		if (bone_idx != -1) {
			sk->connect("bone_pose_changed", callable_mp(this, &BoneAttachment3D::on_bone_pose_update));
			set_transform(sk->get_bone_global_pose(bone_idx));
			bound = true;
		}
	}
}

void BoneAttachment3D::_check_unbind() {
	if (bound) {
		Skeleton3D *sk = Object::cast_to<Skeleton3D>(get_parent());
		if (sk) {
			sk->disconnect("bone_pose_changed", callable_mp(this, &BoneAttachment3D::on_bone_pose_update));
		}
		bound = false;
	}
}

void BoneAttachment3D::_transform_changed() {
	if (override_pose) {
		Skeleton3D *sk = Object::cast_to<Skeleton3D>(get_parent());
		ERR_FAIL_COND_MSG(!sk, "Cannot override pose: Skeleton not found!");
		ERR_FAIL_INDEX_MSG(bone_idx, sk->get_bone_count(), "Cannot override pose: Bone index is out of range!");

		if (override_mode == OVERRIDE_MODES::MODE_GLOBAL_POSE) {
			sk->set_bone_global_pose_override(bone_idx, get_transform(), 1.0, true);
		} else if (override_mode == OVERRIDE_MODES::MODE_LOCAL_POSE) {
			sk->set_bone_local_pose_override(bone_idx, sk->global_pose_to_local_pose(bone_idx, get_transform()), 1.0, true);
		} else if (override_mode == OVERRIDE_MODES::MODE_CUSTOM_POSE) {
			sk->set_bone_custom_pose(bone_idx, sk->global_pose_to_local_pose(bone_idx, get_transform()));
		}
	}
}

void BoneAttachment3D::set_bone_name(const String &p_name) {
	bone_name = p_name;
	Skeleton3D *sk = Object::cast_to<Skeleton3D>(get_parent());
	if (sk) {
		set_bone_idx(sk->find_bone(bone_name));
	}
}

String BoneAttachment3D::get_bone_name() const {
	return bone_name;
}

void BoneAttachment3D::set_bone_idx(const int &p_idx) {
	if (is_inside_tree()) {
		_check_unbind();
	}

	bone_idx = p_idx;

	Skeleton3D *sk = Object::cast_to<Skeleton3D>(get_parent());
	if (sk) {
		if (bone_idx <= -1 || bone_idx >= sk->get_bone_count()) {
			WARN_PRINT("Bone index out of range! Cannot connect BoneAttachment to node!");
			bone_idx = -1;
		} else {
			bone_name = sk->get_bone_name(bone_idx);
		}
	}

	if (is_inside_tree()) {
		_check_bind();
	}

	_change_notify();
}

int BoneAttachment3D::get_bone_idx() const {
	return bone_idx;
}

void BoneAttachment3D::set_override_pose(bool p_override) {
	override_pose = p_override;
	set_notify_local_transform(override_pose);

	if (!override_pose) {
		Skeleton3D *sk = Object::cast_to<Skeleton3D>(get_parent());
		if (sk) {
			if (override_mode == OVERRIDE_MODES::MODE_GLOBAL_POSE) {
				sk->set_bone_global_pose_override(bone_idx, get_transform(), 0.0, false);
			} else if (override_mode == OVERRIDE_MODES::MODE_LOCAL_POSE) {
				sk->set_bone_local_pose_override(bone_idx, get_transform(), 0.0, false);
			} else if (override_mode == OVERRIDE_MODES::MODE_CUSTOM_POSE) {
				sk->set_bone_custom_pose(bone_idx, Transform());
			}
		}
		_transform_changed();
	}
}

bool BoneAttachment3D::get_override_pose() const {
	return override_pose;
}

void BoneAttachment3D::set_override_mode(int p_mode) {
	if (override_pose) {
		Skeleton3D *sk = Object::cast_to<Skeleton3D>(get_parent());
		if (sk) {
			if (override_mode == OVERRIDE_MODES::MODE_GLOBAL_POSE) {
				sk->set_bone_global_pose_override(bone_idx, get_transform(), 0.0, false);
			} else if (override_mode == OVERRIDE_MODES::MODE_LOCAL_POSE) {
				sk->set_bone_local_pose_override(bone_idx, get_transform(), 0.0, false);
			} else if (override_mode == OVERRIDE_MODES::MODE_CUSTOM_POSE) {
				sk->set_bone_custom_pose(bone_idx, Transform());
			}
		}
		override_mode = p_mode;
		_transform_changed();
		return;
	}
	override_mode = p_mode;
}

int BoneAttachment3D::get_override_mode() const {
	return override_mode;
}

void BoneAttachment3D::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE: {
			_check_bind();
		} break;
		case NOTIFICATION_EXIT_TREE: {
			_check_unbind();
		} break;
		case NOTIFICATION_LOCAL_TRANSFORM_CHANGED: {
			_transform_changed();
		} break;
	}
}

void BoneAttachment3D::on_bone_pose_update(int p_bone_index) {
	if (bone_idx == p_bone_index) {
		Skeleton3D *sk = Object::cast_to<Skeleton3D>(get_parent());
		if (sk) {
			if (!override_pose) {
				set_transform(sk->get_bone_global_pose(p_bone_index));
			} else {
				set_notify_local_transform(false);
				if (override_mode == OVERRIDE_MODES::MODE_LOCAL_POSE) {
					set_transform(sk->local_pose_to_global_pose(bone_idx, sk->get_bone_local_pose_override(bone_idx)));
				} else if (override_mode == OVERRIDE_MODES::MODE_CUSTOM_POSE) {
					set_transform(sk->local_pose_to_global_pose(bone_idx, sk->get_bone_custom_pose(bone_idx)));
				}
				set_notify_local_transform(true);
			}
		}
	}
}

BoneAttachment3D::BoneAttachment3D() {
}

void BoneAttachment3D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_bone_name", "bone_name"), &BoneAttachment3D::set_bone_name);
	ClassDB::bind_method(D_METHOD("get_bone_name"), &BoneAttachment3D::get_bone_name);

	ClassDB::bind_method(D_METHOD("set_bone_idx", "bone_idx"), &BoneAttachment3D::set_bone_idx);
	ClassDB::bind_method(D_METHOD("get_bone_idx"), &BoneAttachment3D::get_bone_idx);

	ClassDB::bind_method(D_METHOD("on_bone_pose_update"), &BoneAttachment3D::on_bone_pose_update);

	ClassDB::bind_method(D_METHOD("set_override_pose", "override_pose"), &BoneAttachment3D::set_override_pose);
	ClassDB::bind_method(D_METHOD("get_override_pose"), &BoneAttachment3D::get_override_pose);
	ClassDB::bind_method(D_METHOD("set_override_mode", "override_mode"), &BoneAttachment3D::set_override_mode);
	ClassDB::bind_method(D_METHOD("get_override_mode"), &BoneAttachment3D::get_override_mode);

	ADD_PROPERTY(PropertyInfo(Variant::STRING_NAME, "bone_name"), "set_bone_name", "get_bone_name");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "bone_idx"), "set_bone_idx", "get_bone_idx");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "override_pose"), "set_override_pose", "get_override_pose");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "override_mode", PROPERTY_HINT_ENUM, "Global Pose Override, Local Pose Override, Custom Pose"), "set_override_mode", "get_override_mode");
}
