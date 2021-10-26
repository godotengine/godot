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
		// Because it is a constant function, we cannot use the _get_skeleton_3d function.
		const Skeleton3D *parent = nullptr;
		if (use_external_skeleton) {
			if (external_skeleton_node_cache.is_valid()) {
				parent = Object::cast_to<Skeleton3D>(ObjectDB::get_instance(external_skeleton_node_cache));
			}
		} else {
			parent = Object::cast_to<Skeleton3D>(get_parent());
		}

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

	Node3D::_validate_property(property);
}

bool BoneAttachment3D::_set(const StringName &p_path, const Variant &p_value) {
	if (p_path == SNAME("override_pose")) {
		set_override_pose(p_value);
	} else if (p_path == SNAME("override_mode")) {
		set_override_mode(p_value);
	} else if (p_path == SNAME("use_external_skeleton")) {
		set_use_external_skeleton(p_value);
	} else if (p_path == SNAME("external_skeleton")) {
		set_external_skeleton(p_value);
	}

	return true;
}

bool BoneAttachment3D::_get(const StringName &p_path, Variant &r_ret) const {
	if (p_path == SNAME("override_pose")) {
		r_ret = get_override_pose();
	} else if (p_path == SNAME("override_mode")) {
		r_ret = get_override_mode();
	} else if (p_path == SNAME("use_external_skeleton")) {
		r_ret = get_use_external_skeleton();
	} else if (p_path == SNAME("external_skeleton")) {
		r_ret = get_external_skeleton();
	}

	return true;
}

void BoneAttachment3D::_get_property_list(List<PropertyInfo> *p_list) const {
	p_list->push_back(PropertyInfo(Variant::BOOL, "override_pose", PROPERTY_HINT_NONE, ""));
	if (override_pose) {
		p_list->push_back(PropertyInfo(Variant::INT, "override_mode", PROPERTY_HINT_ENUM, "Global Pose Override, Local Pose Override, Custom Pose"));
	}

	p_list->push_back(PropertyInfo(Variant::BOOL, "use_external_skeleton", PROPERTY_HINT_NONE, ""));
	if (use_external_skeleton) {
		p_list->push_back(PropertyInfo(Variant::NODE_PATH, "external_skeleton", PROPERTY_HINT_NODE_PATH_VALID_TYPES, "Skeleton3D"));
	}
}

TypedArray<String> BoneAttachment3D::get_configuration_warnings() const {
	TypedArray<String> warnings = Node3D::get_configuration_warnings();

	if (use_external_skeleton) {
		if (external_skeleton_node_cache.is_null()) {
			warnings.append(TTR("External Skeleton3D node not set! Please set a path to an external Skeleton3D node."));
		}
	} else {
		Skeleton3D *parent = Object::cast_to<Skeleton3D>(get_parent());
		if (!parent) {
			warnings.append(TTR("Parent node is not a Skeleton3D node! Please use an external Skeleton3D if you intend to use the BoneAttachment3D without it being a child of a Skeleton3D node."));
		}
	}

	if (bone_idx == -1) {
		warnings.append(TTR("BoneAttachment3D node is not bound to any bones! Please select a bone to attach this node."));
	}

	return warnings;
}

void BoneAttachment3D::_update_external_skeleton_cache() {
	external_skeleton_node_cache = ObjectID();
	if (has_node(external_skeleton_node)) {
		Node *node = get_node(external_skeleton_node);
		ERR_FAIL_COND_MSG(!node, "Cannot update external skeleton cache: Node cannot be found!");

		// Make sure it's a skeleton3D
		Skeleton3D *sk = Object::cast_to<Skeleton3D>(node);
		ERR_FAIL_COND_MSG(!sk, "Cannot update external skeleton cache: Skeleton3D Nodepath does not point to a Skeleton3D node!");

		external_skeleton_node_cache = node->get_instance_id();
	} else {
		if (external_skeleton_node.is_empty()) {
			BoneAttachment3D *parent_attachment = Object::cast_to<BoneAttachment3D>(get_parent());
			if (parent_attachment) {
				parent_attachment->_update_external_skeleton_cache();
				if (parent_attachment->has_node(parent_attachment->external_skeleton_node)) {
					Node *node = parent_attachment->get_node(parent_attachment->external_skeleton_node);
					ERR_FAIL_COND_MSG(!node, "Cannot update external skeleton cache: Parent's Skeleton3D node cannot be found!");

					// Make sure it's a skeleton3D
					Skeleton3D *sk = Object::cast_to<Skeleton3D>(node);
					ERR_FAIL_COND_MSG(!sk, "Cannot update external skeleton cache: Parent Skeleton3D Nodepath does not point to a Skeleton3D node!");

					external_skeleton_node_cache = node->get_instance_id();
					external_skeleton_node = get_path_to(node);
				}
			}
		}
	}
}

void BoneAttachment3D::_check_bind() {
	Skeleton3D *sk = _get_skeleton3d();

	if (sk && !bound) {
		if (bone_idx <= -1) {
			bone_idx = sk->find_bone(bone_name);
		}
		if (bone_idx != -1) {
			sk->call_deferred(SNAME("connect"), "bone_pose_changed", callable_mp(this, &BoneAttachment3D::on_bone_pose_update));
			bound = true;
			call_deferred(SNAME("on_bone_pose_update"), bone_idx);
		}
	}
}

Skeleton3D *BoneAttachment3D::_get_skeleton3d() {
	if (use_external_skeleton) {
		if (external_skeleton_node_cache.is_valid()) {
			return Object::cast_to<Skeleton3D>(ObjectDB::get_instance(external_skeleton_node_cache));
		} else {
			_update_external_skeleton_cache();
			if (external_skeleton_node_cache.is_valid()) {
				return Object::cast_to<Skeleton3D>(ObjectDB::get_instance(external_skeleton_node_cache));
			}
		}
	} else {
		return Object::cast_to<Skeleton3D>(get_parent());
	}
	return nullptr;
}

void BoneAttachment3D::_check_unbind() {
	if (bound) {
		Skeleton3D *sk = _get_skeleton3d();

		if (sk) {
			sk->disconnect(SNAME("bone_pose_changed"), callable_mp(this, &BoneAttachment3D::on_bone_pose_update));
		}
		bound = false;
	}
}

void BoneAttachment3D::_transform_changed() {
	if (!is_inside_tree()) {
		return;
	}

	if (override_pose) {
		Skeleton3D *sk = _get_skeleton3d();

		ERR_FAIL_COND_MSG(!sk, "Cannot override pose: Skeleton not found!");
		ERR_FAIL_INDEX_MSG(bone_idx, sk->get_bone_count(), "Cannot override pose: Bone index is out of range!");

		Transform3D our_trans = get_transform();
		if (use_external_skeleton) {
			our_trans = sk->world_transform_to_global_pose(get_global_transform());
		}

		if (override_mode == OVERRIDE_MODES::MODE_GLOBAL_POSE) {
			sk->set_bone_global_pose_override(bone_idx, our_trans, 1.0, true);
		} else if (override_mode == OVERRIDE_MODES::MODE_LOCAL_POSE) {
			sk->set_bone_local_pose_override(bone_idx, sk->global_pose_to_local_pose(bone_idx, our_trans), 1.0, true);
		}
	}
}

void BoneAttachment3D::set_bone_name(const String &p_name) {
	bone_name = p_name;
	Skeleton3D *sk = _get_skeleton3d();
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

	Skeleton3D *sk = _get_skeleton3d();
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

	notify_property_list_changed();
}

int BoneAttachment3D::get_bone_idx() const {
	return bone_idx;
}

void BoneAttachment3D::set_override_pose(bool p_override) {
	override_pose = p_override;
	set_notify_local_transform(override_pose);
	set_process_internal(override_pose);

	if (!override_pose) {
		Skeleton3D *sk = _get_skeleton3d();
		if (sk) {
			if (override_mode == OVERRIDE_MODES::MODE_GLOBAL_POSE) {
				sk->set_bone_global_pose_override(bone_idx, Transform3D(), 0.0, false);
			} else if (override_mode == OVERRIDE_MODES::MODE_LOCAL_POSE) {
				sk->set_bone_local_pose_override(bone_idx, Transform3D(), 0.0, false);
			}
		}
		_transform_changed();
	}
	notify_property_list_changed();
}

bool BoneAttachment3D::get_override_pose() const {
	return override_pose;
}

void BoneAttachment3D::set_override_mode(int p_mode) {
	if (override_pose) {
		Skeleton3D *sk = _get_skeleton3d();
		if (sk) {
			if (override_mode == OVERRIDE_MODES::MODE_GLOBAL_POSE) {
				sk->set_bone_global_pose_override(bone_idx, Transform3D(), 0.0, false);
			} else if (override_mode == OVERRIDE_MODES::MODE_LOCAL_POSE) {
				sk->set_bone_local_pose_override(bone_idx, Transform3D(), 0.0, false);
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

void BoneAttachment3D::set_use_external_skeleton(bool p_use_external) {
	use_external_skeleton = p_use_external;

	if (use_external_skeleton) {
		_check_unbind();
		_update_external_skeleton_cache();
		_check_bind();
		_transform_changed();
	}

	notify_property_list_changed();
}

bool BoneAttachment3D::get_use_external_skeleton() const {
	return use_external_skeleton;
}

void BoneAttachment3D::set_external_skeleton(NodePath p_path) {
	external_skeleton_node = p_path;
	_update_external_skeleton_cache();
	notify_property_list_changed();
}

NodePath BoneAttachment3D::get_external_skeleton() const {
	return external_skeleton_node;
}

void BoneAttachment3D::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE: {
			if (use_external_skeleton) {
				_update_external_skeleton_cache();
			}
			_check_bind();
		} break;
		case NOTIFICATION_EXIT_TREE: {
			_check_unbind();
		} break;
		case NOTIFICATION_LOCAL_TRANSFORM_CHANGED: {
			_transform_changed();
		} break;
		case NOTIFICATION_INTERNAL_PROCESS: {
			if (_override_dirty) {
				_override_dirty = false;
			}
		}
	}
}

void BoneAttachment3D::on_bone_pose_update(int p_bone_index) {
	if (bone_idx == p_bone_index) {
		Skeleton3D *sk = _get_skeleton3d();
		if (sk) {
			if (!override_pose) {
				if (use_external_skeleton) {
					set_global_transform(sk->global_pose_to_world_transform(sk->get_bone_global_pose(bone_idx)));
				} else {
					set_transform(sk->get_bone_global_pose(bone_idx));
				}
			} else {
				if (!_override_dirty) {
					_transform_changed();
					_override_dirty = true;
				}
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

	ClassDB::bind_method(D_METHOD("on_bone_pose_update", "bone_index"), &BoneAttachment3D::on_bone_pose_update);

	ClassDB::bind_method(D_METHOD("set_override_pose", "override_pose"), &BoneAttachment3D::set_override_pose);
	ClassDB::bind_method(D_METHOD("get_override_pose"), &BoneAttachment3D::get_override_pose);
	ClassDB::bind_method(D_METHOD("set_override_mode", "override_mode"), &BoneAttachment3D::set_override_mode);
	ClassDB::bind_method(D_METHOD("get_override_mode"), &BoneAttachment3D::get_override_mode);

	ClassDB::bind_method(D_METHOD("set_use_external_skeleton", "use_external_skeleton"), &BoneAttachment3D::set_use_external_skeleton);
	ClassDB::bind_method(D_METHOD("get_use_external_skeleton"), &BoneAttachment3D::get_use_external_skeleton);
	ClassDB::bind_method(D_METHOD("set_external_skeleton", "external_skeleton"), &BoneAttachment3D::set_external_skeleton);
	ClassDB::bind_method(D_METHOD("get_external_skeleton"), &BoneAttachment3D::get_external_skeleton);

	ADD_PROPERTY(PropertyInfo(Variant::STRING_NAME, "bone_name"), "set_bone_name", "get_bone_name");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "bone_idx"), "set_bone_idx", "get_bone_idx");
}
