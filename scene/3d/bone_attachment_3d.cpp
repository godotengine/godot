/**************************************************************************/
/*  bone_attachment_3d.cpp                                                */
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

#include "bone_attachment_3d.h"
#include "bone_attachment_3d.compat.inc"

#include "scene/3d/skeleton_modifier_3d.h"

void BoneAttachment3D::_validate_property(PropertyInfo &p_property) const {
	if (p_property.name == "bone_name") {
		// Because it is a constant function, we cannot use the get_skeleton function.
		const Skeleton3D *parent = nullptr;
		if (use_external_skeleton) {
			if (external_skeleton_node_cache.is_valid()) {
				parent = ObjectDB::get_instance<Skeleton3D>(external_skeleton_node_cache);
			}
		} else {
			parent = Object::cast_to<Skeleton3D>(get_parent());
		}

		if (parent) {
			p_property.hint = PROPERTY_HINT_ENUM;
			p_property.hint_string = parent->get_concatenated_bone_names();
		} else {
			p_property.hint = PROPERTY_HINT_NONE;
			p_property.hint_string = "";
		}
	}

	if (p_property.name == "external_skeleton" && !use_external_skeleton) {
		p_property.usage = PROPERTY_USAGE_NONE;
	}
	if (override_pose) {
		if (p_property.name == "subscribe_modifier" || p_property.name == "extra_update") {
			p_property.usage = PROPERTY_USAGE_NONE;
		}
	} else if (extra_update != EXTRA_UPDATE_SUBSCRIBE_MODIFIER) {
		if (p_property.name == "subscribe_modifier") {
			p_property.usage = PROPERTY_USAGE_NONE;
		}
	}
}

PackedStringArray BoneAttachment3D::get_configuration_warnings() const {
	PackedStringArray warnings = Node3D::get_configuration_warnings();

	if (use_external_skeleton) {
		if (external_skeleton_node_cache.is_null()) {
			warnings.push_back(RTR("External Skeleton3D node not set! Please set a path to an external Skeleton3D node."));
		}
	} else {
		Skeleton3D *parent = Object::cast_to<Skeleton3D>(get_parent());
		if (!parent) {
			warnings.push_back(RTR("Parent node is not a Skeleton3D node! Please use an external Skeleton3D if you intend to use the BoneAttachment3D without it being a child of a Skeleton3D node."));
		}
	}

	if (bone_idx == -1) {
		warnings.push_back(RTR("BoneAttachment3D node is not bound to any bones! Please select a bone to attach this node."));
	}

	return warnings;
}

void BoneAttachment3D::_update_node_cache() {
	_update_external_skeleton_cache();
	_update_subscribe_node_cache();
}

void BoneAttachment3D::_update_external_skeleton_cache() {
	external_skeleton_node_cache = ObjectID();
	if (has_node(external_skeleton_node)) {
		Node *node = get_node(external_skeleton_node);
		ERR_FAIL_NULL_MSG(node, "Cannot update external skeleton cache: Node cannot be found!");

		// Make sure it's a skeleton3D.
		Skeleton3D *sk = Object::cast_to<Skeleton3D>(node);
		ERR_FAIL_NULL_MSG(sk, "Cannot update external skeleton cache: Skeleton3D Nodepath does not point to a Skeleton3D node!");

		external_skeleton_node_cache = node->get_instance_id();
	} else {
		if (external_skeleton_node.is_empty()) {
			BoneAttachment3D *parent_attachment = Object::cast_to<BoneAttachment3D>(get_parent());
			if (parent_attachment) {
				parent_attachment->_update_external_skeleton_cache();
				if (parent_attachment->has_node(parent_attachment->external_skeleton_node)) {
					Node *node = parent_attachment->get_node(parent_attachment->external_skeleton_node);
					ERR_FAIL_NULL_MSG(node, "Cannot update external skeleton cache: Parent's Skeleton3D node cannot be found!");

					// Make sure it's a skeleton3D.
					Skeleton3D *sk = Object::cast_to<Skeleton3D>(node);
					ERR_FAIL_NULL_MSG(sk, "Cannot update external skeleton cache: Parent Skeleton3D Nodepath does not point to a Skeleton3D node!");

					external_skeleton_node_cache = node->get_instance_id();
					external_skeleton_node = get_path_to(node);
				}
			}
		}
	}
}

void BoneAttachment3D::_update_subscribe_node_cache() {
	subscribe_node_cache = ObjectID();
	if (extra_update == EXTRA_UPDATE_BEFORE_SKELETON_UPDATE) {
		Skeleton3D *sk = get_skeleton();
		ERR_FAIL_NULL_MSG(sk, "Cannot update subscribe node cache: Node cannot be found!");
		if (sk) {
			subscribe_node_cache = sk->get_instance_id();
		}
	} else if (extra_update == EXTRA_UPDATE_SUBSCRIBE_MODIFIER) {
		if (has_node(subscribe_modifier)) {
			Node *node = get_node(subscribe_modifier);
			ERR_FAIL_NULL_MSG(node, "Cannot update subscribe node cache: Node cannot be found!");

			// Make sure it's a SkeletonModifier3D.
			SkeletonModifier3D *sm = Object::cast_to<SkeletonModifier3D>(node);
			if (sm) {
				subscribe_node_cache = node->get_instance_id();
				if (sm->get_skeleton() != get_skeleton()) {
					WARN_PRINT_ED("BoneAttachment3D: The SkeletonModifier3D node '" + sm->get_name() + "' is not attached to the same Skeleton3D as this BoneAttachment3D node. This may lead to unexpected behavior.");
				}
			}
		}
	}
}

void BoneAttachment3D::_check_bind() {
	_check_bind_subscribe();
	_check_bind_skeleton();
}

void BoneAttachment3D::_check_bind_skeleton() {
	if (bound_skeleton) {
		return;
	}
	Skeleton3D *sk = get_skeleton();
	if (sk) {
		ERR_FAIL_NULL_MSG(sk, "Cannot connect update signal: Node cannot be found!");
		if (bone_idx <= -1) {
			bone_idx = sk->find_bone(bone_name);
		}
		if (bone_idx != -1) {
			sk->connect(SceneStringName(skeleton_updated), callable_mp(this, &BoneAttachment3D::on_skeleton_update));
			bound_skeleton = true;
			on_skeleton_update();
		}
	}
}

void BoneAttachment3D::_check_bind_subscribe() {
	if (bound_subscribe || override_pose) {
		return;
	}
	Node *node = ObjectDB::get_instance<Node>(subscribe_node_cache);
	if (node) {
		Skeleton3D *sk = Object::cast_to<Skeleton3D>(node);
		if (sk) {
			sk->connect(SNAME("skeleton_update_started"), callable_mp(this, &BoneAttachment3D::on_skeleton_update));
			bound_subscribe = true;
		}
		SkeletonModifier3D *sm = Object::cast_to<SkeletonModifier3D>(node);
		if (sm) {
			sm->connect(SNAME("modification_processed"), callable_mp(this, &BoneAttachment3D::on_skeleton_update));
			bound_subscribe = true;
		}
	}
}

void BoneAttachment3D::_check_unbind() {
	_check_unbind_subscribe();
	_check_unbind_skeleton();
}

void BoneAttachment3D::_check_unbind_skeleton() {
	if (!bound_skeleton) {
		return;
	}
	Skeleton3D *sk = get_skeleton();
	if (sk) {
		sk->disconnect(SceneStringName(skeleton_updated), callable_mp(this, &BoneAttachment3D::on_skeleton_update));
	}
	bound_skeleton = false;
}

void BoneAttachment3D::_check_unbind_subscribe() {
	if (!bound_subscribe) {
		return;
	}
	Node *node = ObjectDB::get_instance<Node>(subscribe_node_cache);
	ERR_FAIL_NULL_MSG(node, "Cannot disconnect update signal: Node cannot be found!");
	if (node) {
		Skeleton3D *sk = Object::cast_to<Skeleton3D>(node);
		if (sk) {
			sk->disconnect(SNAME("skeleton_update_started"), callable_mp(this, &BoneAttachment3D::on_skeleton_update));
		}
		SkeletonModifier3D *sm = Object::cast_to<SkeletonModifier3D>(node);
		if (sm) {
			sm->disconnect(SNAME("modification_processed"), callable_mp(this, &BoneAttachment3D::on_skeleton_update));
		}
	}
	bound_subscribe = false;
}

void BoneAttachment3D::_transform_changed() {
	if (!is_inside_tree()) {
		return;
	}

	if (override_pose && !overriding) {
		Skeleton3D *sk = get_skeleton();

		ERR_FAIL_NULL_MSG(sk, "Cannot override pose: Skeleton not found!");
		ERR_FAIL_INDEX_MSG(bone_idx, sk->get_bone_count(), "Cannot override pose: Bone index is out of range!");

		Transform3D our_trans = get_transform();
		if (use_external_skeleton) {
			our_trans = sk->get_global_transform().affine_inverse() * get_global_transform();
		}

		overriding = true;
		sk->set_bone_global_pose(bone_idx, our_trans);
		sk->force_update_all_dirty_bones();
	}
	overriding = false;
}

void BoneAttachment3D::_validate_bind_states() {
	if (!is_inside_tree()) {
		return;
	}
	_check_unbind();
	_update_node_cache();
	_check_bind();
	_transform_changed();
}

Skeleton3D *BoneAttachment3D::get_skeleton() {
	if (use_external_skeleton) {
		if (external_skeleton_node_cache.is_valid()) {
			return ObjectDB::get_instance<Skeleton3D>(external_skeleton_node_cache);
		} else {
			_update_external_skeleton_cache();
			if (external_skeleton_node_cache.is_valid()) {
				return ObjectDB::get_instance<Skeleton3D>(external_skeleton_node_cache);
			}
		}
	} else {
		return Object::cast_to<Skeleton3D>(get_parent());
	}
	return nullptr;
}

void BoneAttachment3D::set_bone_name(const String &p_name) {
	bone_name = p_name;
	Skeleton3D *sk = get_skeleton();
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

	Skeleton3D *sk = get_skeleton();
	if (sk) {
		if (bone_idx <= -1 || bone_idx >= sk->get_bone_count()) {
			WARN_PRINT("Bone index " + itos(bone_idx) + " out of range! Cannot connect BoneAttachment to node!");
			bone_idx = -1;
		} else {
			bone_name = sk->get_bone_name(bone_idx);
		}
	}

	if (is_inside_tree()) {
		_check_bind();
	} else {
		on_skeleton_update();
	}

	notify_property_list_changed();
}

int BoneAttachment3D::get_bone_idx() const {
	return bone_idx;
}

void BoneAttachment3D::set_override_pose(bool p_override) {
	if (override_pose == p_override) {
		return;
	}

	override_pose = p_override;
	set_notify_transform(override_pose);
	set_process_internal(override_pose);
	if (!override_pose && bone_idx >= 0) {
		Skeleton3D *sk = get_skeleton();
		if (sk) {
			sk->reset_bone_pose(bone_idx);
		}
	}

	if (is_inside_tree()) {
		if (override_pose) {
			_check_unbind_subscribe();
		} else {
			_check_bind_subscribe();
		}
	}

	notify_property_list_changed();
}

bool BoneAttachment3D::get_override_pose() const {
	return override_pose;
}

void BoneAttachment3D::set_use_external_skeleton(bool p_use_external_skeleton) {
	use_external_skeleton = p_use_external_skeleton;

	_validate_bind_states();
	notify_property_list_changed();
}

bool BoneAttachment3D::get_use_external_skeleton() const {
	return use_external_skeleton;
}

void BoneAttachment3D::set_external_skeleton(NodePath p_external_skeleton) {
	external_skeleton_node = p_external_skeleton;

	_validate_bind_states();
	notify_property_list_changed();
}

NodePath BoneAttachment3D::get_external_skeleton() const {
	return external_skeleton_node;
}

void BoneAttachment3D::set_extra_update(ExtraUpdate p_extra_update) {
	if (override_pose) {
		return;
	}
	extra_update = p_extra_update;

	_validate_bind_states();
	notify_property_list_changed();
}

BoneAttachment3D::ExtraUpdate BoneAttachment3D::get_extra_update() const {
	return extra_update;
}

void BoneAttachment3D::set_subscribe_modifier(NodePath p_subscribe_modifier) {
	if (override_pose || extra_update != EXTRA_UPDATE_SUBSCRIBE_MODIFIER) {
		return;
	}
	subscribe_modifier = p_subscribe_modifier;

	_validate_bind_states();
}

NodePath BoneAttachment3D::get_subscribe_modifier() const {
	return subscribe_modifier;
}

void BoneAttachment3D::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE: {
			_update_node_cache();
			_check_bind();
		} break;

		case NOTIFICATION_EXIT_TREE: {
			_check_unbind();
		} break;

		case NOTIFICATION_TRANSFORM_CHANGED: {
			_transform_changed();
		} break;

		case NOTIFICATION_INTERNAL_PROCESS: {
			if (_override_dirty) {
				_override_dirty = false;
			}
		} break;
	}
}

void BoneAttachment3D::on_skeleton_update() {
	if (updating) {
		return;
	}
	updating = true;
	if (bone_idx >= 0) {
		Skeleton3D *sk = get_skeleton();
		if (sk) {
			if (!override_pose) {
				if (use_external_skeleton) {
					set_global_transform(sk->get_global_transform() * sk->get_bone_global_pose(bone_idx));
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
	updating = false;
}

#ifdef TOOLS_ENABLED
void BoneAttachment3D::notify_skeleton_bones_renamed(Node *p_base_scene, Skeleton3D *p_skeleton, Dictionary p_rename_map) {
	const Skeleton3D *parent = nullptr;
	if (use_external_skeleton) {
		if (external_skeleton_node_cache.is_valid()) {
			parent = ObjectDB::get_instance<Skeleton3D>(external_skeleton_node_cache);
		}
	} else {
		parent = Object::cast_to<Skeleton3D>(get_parent());
	}
	if (parent && parent == p_skeleton) {
		StringName bn = p_rename_map[bone_name];
		if (bn) {
			set_bone_name(bn);
		}
	}
}

void BoneAttachment3D::notify_rebind_required() {
	// Ensures bindings are properly updated after a scene reload.
	_check_unbind();
	if (use_external_skeleton) {
		_update_external_skeleton_cache();
	}
	bone_idx = -1;
	_update_node_cache();
	_check_bind();
}
#endif // TOOLS_ENABLED

BoneAttachment3D::BoneAttachment3D() {
	set_physics_interpolation_mode(PHYSICS_INTERPOLATION_MODE_OFF);
}

void BoneAttachment3D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_skeleton"), &BoneAttachment3D::get_skeleton);

	ClassDB::bind_method(D_METHOD("set_bone_name", "bone_name"), &BoneAttachment3D::set_bone_name);
	ClassDB::bind_method(D_METHOD("get_bone_name"), &BoneAttachment3D::get_bone_name);

	ClassDB::bind_method(D_METHOD("set_bone_idx", "bone_idx"), &BoneAttachment3D::set_bone_idx);
	ClassDB::bind_method(D_METHOD("get_bone_idx"), &BoneAttachment3D::get_bone_idx);

	ClassDB::bind_method(D_METHOD("on_skeleton_update"), &BoneAttachment3D::on_skeleton_update);

	ClassDB::bind_method(D_METHOD("set_override_pose", "override_pose"), &BoneAttachment3D::set_override_pose);
	ClassDB::bind_method(D_METHOD("get_override_pose"), &BoneAttachment3D::get_override_pose);

	ClassDB::bind_method(D_METHOD("set_use_external_skeleton", "use_external_skeleton"), &BoneAttachment3D::set_use_external_skeleton);
	ClassDB::bind_method(D_METHOD("get_use_external_skeleton"), &BoneAttachment3D::get_use_external_skeleton);
	ClassDB::bind_method(D_METHOD("set_external_skeleton", "external_skeleton"), &BoneAttachment3D::set_external_skeleton);
	ClassDB::bind_method(D_METHOD("get_external_skeleton"), &BoneAttachment3D::get_external_skeleton);

	ClassDB::bind_method(D_METHOD("set_extra_update", "extra_update"), &BoneAttachment3D::set_extra_update);
	ClassDB::bind_method(D_METHOD("get_extra_update"), &BoneAttachment3D::get_extra_update);
	ClassDB::bind_method(D_METHOD("set_subscribe_modifier", "subscribe_modifier"), &BoneAttachment3D::set_subscribe_modifier);
	ClassDB::bind_method(D_METHOD("get_subscribe_modifier"), &BoneAttachment3D::get_subscribe_modifier);

	ADD_PROPERTY(PropertyInfo(Variant::STRING_NAME, "bone_name"), "set_bone_name", "get_bone_name");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "bone_idx"), "set_bone_idx", "get_bone_idx");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "override_pose"), "set_override_pose", "get_override_pose");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "extra_update", PROPERTY_HINT_ENUM, "None,Before Skeleton Update,Subscribe Modifier"), "set_extra_update", "get_extra_update");
	ADD_PROPERTY(PropertyInfo(Variant::NODE_PATH, "subscribe_modifier", PROPERTY_HINT_NODE_PATH_VALID_TYPES, "SkeletonModifier3D"), "set_subscribe_modifier", "get_subscribe_modifier");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "use_external_skeleton"), "set_use_external_skeleton", "get_use_external_skeleton");
	ADD_PROPERTY(PropertyInfo(Variant::NODE_PATH, "external_skeleton", PROPERTY_HINT_NODE_PATH_VALID_TYPES, "Skeleton3D"), "set_external_skeleton", "get_external_skeleton");

	BIND_ENUM_CONSTANT(EXTRA_UPDATE_NONE);
	BIND_ENUM_CONSTANT(EXTRA_UPDATE_BEFORE_SKELETON_UPDATE);
	BIND_ENUM_CONSTANT(EXTRA_UPDATE_SUBSCRIBE_MODIFIER);
}
